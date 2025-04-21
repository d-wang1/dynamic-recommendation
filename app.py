import asyncio, os
# 1) ensure there is always an event loop
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# 2) disable the file watcher that trips over torch._classes
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"]      = "false"

import streamlit as st
import torch
import pandas as pd
from pathlib import Path

from data.helper import (
    load_config,
    load_movies,
    load_ratings,
    gen_demographic_table,
    load_save_neumf_table,
)
from models.hyper_neumf import DCHyperNeuMF
from data.objs import AGE_BUCKETS, OCCUPATIONS, AGE2IDX
# --- 1. Setup & caching -------------------------------------------------
@st.cache_resource
def setup_everything(config_path=None):
    # Load config
    cfg = load_config(config_path)

    # Extract and save NeuMF movie tables (GMF + MLP)
    item_gmf, item_mlp = load_save_neumf_table(config_path, verbose=cfg.get("verbose", False))

    # Build DC-HyperNeuMF model from checkpoint
    dc_ckpt = Path(cfg["app"]["ckpt_to_use"]).expanduser().resolve()
    model: DCHyperNeuMF = DCHyperNeuMF.load_from_checkpoint(str(dc_ckpt))
    # Overwrite movie embeddings and freeze them
    model.item_gmf.weight.data.copy_(item_gmf)
    model.item_mlp.weight.data.copy_(item_mlp)
    for p in list(model.item_gmf.parameters()) + list(model.item_mlp.parameters()):
        p.requires_grad_(False)
    model.eval()

    # Load internal ratings + demographics
    ratings_int, demog_tensor, _, n_items = gen_demographic_table(
        config_path=config_path, verbose=cfg.get("verbose", False)
    )

    # Build raw->internal MovieID mapping from raw ratings
    raw_ratings = load_ratings(config_path=config_path, verbose=cfg.get("verbose", False))
    mid_map = {m: i for i, m in enumerate(raw_ratings.MovieID.unique())}

    # Load movie titles and align to internal IDs
    movies_df = load_movies(config_path=config_path, verbose=cfg.get("verbose", False))
    titles = [None] * n_items
    for _, row in movies_df.iterrows():
        internal = mid_map.get(row.MovieID)
        if internal is not None:
            titles[internal] = row.Title



    gm = ratings_int.groupby("mid")["Rating"].mean().reset_index()
    global_means = torch.zeros(n_items)
    for _, r in gm.iterrows():
        global_means[int(r.mid)] = r.Rating


    return model, ratings_int, demog_tensor, titles, global_means

# Initialize everything
model, ratings_df, demog_tensor, titles, global_means = setup_everything("config.json")


# --- 2. Streamlit GUI ----------------------------------------------------
st.title("🎬 Cold-Start Recommender Demo")

mode = st.sidebar.radio("Mode", ["Existing User", "New User"])
topk = st.sidebar.slider("Top-K", 5, 20, 10)


def vectorized_scores(model, demog_row, supp_m, supp_r):
    with torch.no_grad():
        # 1) user embeddings
        u_gmf, u_mlp = model.make_user_embs(
            demog_row.unsqueeze(0),
            supp_m.unsqueeze(0),
            supp_r.unsqueeze(0),
        )                                   # (1, d_emb) each

        # 2) item tables
        V_gmf   = model.item_gmf.weight     # (N, d_emb)
        V_mlp   = model.item_mlp.weight     # (N, d_emb)
        V_demog = model.item_demog.weight   # (N, demog_feat_dim)

        # 3) branch vectors
        gmf_vec = u_gmf * V_gmf              # (N, d_emb)
        um      = u_mlp.repeat(len(V_mlp), 1)
        mlp_feats = model.mlp_head(torch.cat([um, V_mlp], dim=1))  # (N, mlp_last)

        # demographic vector
        # rebuild dem_feats as in forward (just replicate for all items):
        g = model.gender_emb(demog_row[[0]])
        device = demog_row.device
        # 1) extract the raw age code, e.g. 25, 56, etc.
        raw_age = demog_row[1].item()
        # 2) map to your 0–6 index
        age_idx = AGE2IDX.get(raw_age, 0)  # default to 0 if missing
        # 3) make a 1‑element tensor on the right device
        age_tensor = torch.tensor([age_idx], device=device)
        # 4) lookup embedding
        a = model.age_emb(age_tensor)
        o = model.occ_emb(demog_row[2].unsqueeze(0))
        z = model.zip_emb(demog_row[3].unsqueeze(0))
        dem_feats = torch.cat([g,a,o,z], dim=1)                # (1, demog_feat_dim)
        dem_feats = dem_feats.repeat(len(V_demog), 1)          # (N, demog_feat_dim)
        dem_in    = dem_feats * V_demog                        # (N, demog_feat_dim)

        # 4) branch scores
        s_gmf   = model.out_gmf(   gmf_vec).squeeze(1)        # (N,)
        s_mlp   = model.out_mlp(   mlp_feats).squeeze(1)      # (N,)
        s_demog = model.out_demog(dem_in).squeeze(1)          # (N,)

        # 5) gate weights
        gate_in = torch.cat([gmf_vec, mlp_feats, dem_in], dim=1)  # (N, sum_dims)
        wts     = model.gate3(gate_in)                            # (N,3)

        # 6) final fused scores
        fused = wts[:,0]*s_gmf + wts[:,1]*s_mlp + wts[:,2]*s_demog  # (N,)
        return fused


if mode == "Existing User":
    uid = st.selectbox("Select User ID", sorted(ratings_df.uid.unique()))
    d = demog_tensor[uid]
    st.markdown(f"**Gender:** {'F' if d[0]==1 else 'M'}  ")
    st.markdown(f"**Age bucket:** {int(d[1])}  ")
    st.markdown(f"**Occupation code:** {int(d[2])}  ")
    st.markdown(f"**ZIP prefix:** {int(d[3])}  ")

    # display recent support ratings
    user_hist = ratings_df[ratings_df.uid==uid].sort_values("Timestamp", ascending=False)
    max_k = min(len(user_hist), 10)
    k = st.slider("Support-set size (k)", 1, max_k, 5)
    support = user_hist.head(k)
    st.markdown("#### Support Ratings (edit if you like)")
    support_df = pd.DataFrame({
        "Movie": [titles[mid] for mid in support.mid],
        "Rating": support.Rating.values
    })
    edited = st.data_editor(
        support_df, 
        num_rows="fixed", 
        key="support_editor"
    )
    # Now pull the edits back out:
    supp_m = torch.tensor(
        [titles.index(m) for m in edited["Movie"]],
        dtype=torch.long
    )
    supp_r = torch.tensor(
        edited["Rating"].astype(float).tolist(),
        dtype=torch.float
    )

    if st.button("Recommend"):
        # st.write("Support movie IDs:", supp_m.tolist())
        # st.write("Support ratings:  ", supp_r.tolist())
        # u_gmf, u_mlp = model.make_user_embs(d.unsqueeze(0), supp_m.unsqueeze(0), supp_r.unsqueeze(0))
        # st.write("u_gmf[:5]:", u_gmf[0,:5].tolist())
        # Vectorized scoring using the *edited* support set
        scores = vectorized_scores(model, d, supp_m, supp_r)  # (n_items,)
        vals, idxs = scores.topk(topk)
        # 1) raw predicted scores
        scores = vectorized_scores(model, d, supp_m, supp_r)
        # 2) subtract global popularity → residuals
        residuals = scores - global_means
        # 3) pick Top‑K by residual
        vals, idxs = residuals.topk(topk)



        st.markdown("### Top Recommendations")
        for delta, idx in zip(vals.cpu().tolist(), idxs.cpu().tolist()):
            raw = scores[idx].item()
            st.write(
                f"- **{titles[idx]}**  "
                f"(pred {raw:.2f}, Δ={delta:+.2f})"
            )

else:  # New User
    st.markdown("### Enter New User Profile")

    # — Age bucket dropdown —
    age_key = st.selectbox(
        "Age bucket",
        options=list(AGE_BUCKETS.keys()),
        format_func=lambda x: AGE_BUCKETS[x],
    )

    gender = st.radio("Gender", ["M", "F"])

    # — Occupation dropdown —
    occ_key = st.selectbox(
        "Occupation",
        options=list(OCCUPATIONS.keys()),
        format_func=lambda x: OCCUPATIONS[x],
    )

    zp = st.text_input("ZIP prefix (first 2 digits)", "63")

    # build the demographic vector in the same order your model expects:
    # [gender_id (0=M,1=F), age_bucket_code, occupation_code, zip_prefix]
    d_row = torch.tensor([
        1 if gender == "F" else 0,
        age_key,
        occ_key,
        int(zp)
    ], dtype=torch.long)

    k = st.slider("Number of support ratings (k)", 0, 10, 0)
    supp_m_list, supp_r_list = [], []
    if k > 0:
        st.markdown(f"#### Enter {k} sample ratings")
        for i in range(k):
            movie = st.selectbox(f"Movie #{i+1}", titles, key=f"new_m{i}")
            rating = st.slider(f"Rating for '{movie}'", 1.0, 5.0, 3.0, key=f"new_r{i}")
            supp_m_list.append(titles.index(movie))
            supp_r_list.append(rating)

    supp_m = torch.tensor(supp_m_list, dtype=torch.long)
    supp_r = torch.tensor(supp_r_list, dtype=torch.float)

    if st.button("Recommend"):
        # st.write("Demographics:", {
        #     "Gender": gender,
        #     "Age": AGE_BUCKETS[age_key],
        #     "Occupation": OCCUPATIONS[occ_key],
        #     "ZIP": zp,
        # })
        # st.write("Support set (movie IDs / ratings):", list(zip(supp_m_list, supp_r_list)))

        scores = vectorized_scores(model, d_row, supp_m, supp_r)
        vals, idxs = scores.topk(topk)
        # 1) raw predicted scores
        scores = vectorized_scores(model, d_row, supp_m, supp_r)
        # 2) subtract global popularity → residuals
        residuals = scores - global_means
        # 3) pick Top‑K by residual
        vals, idxs = residuals.topk(topk)



        st.markdown("### Top Recommendations")
        for delta, idx in zip(vals.cpu().tolist(), idxs.cpu().tolist()):
            raw = scores[idx].item()
            st.write(
                f"- **{titles[idx]}**  "
                f"(pred {raw:.2f}, Δ={delta:+.2f})"
            )