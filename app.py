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
import torch.nn.functional as F
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
    cfg = load_config(config_path)
    item_gmf, item_mlp = load_save_neumf_table(
        config_path, verbose=cfg.get("verbose", False)
    )
    dc_ckpt = Path(cfg["app"]["ckpt_to_use"]).expanduser().resolve()
    model: DCHyperNeuMF = DCHyperNeuMF.load_from_checkpoint(str(dc_ckpt))
    model.item_gmf.weight.data.copy_(item_gmf)
    model.item_mlp.weight.data.copy_(item_mlp)
    for p in model.item_gmf.parameters():
        p.requires_grad_(False)
    for p in model.item_mlp.parameters():
        p.requires_grad_(False)
    model.eval()

    ratings_int, demog_tensor, _, n_items = gen_demographic_table(
        config_path=config_path, verbose=cfg.get("verbose", False)
    )

    raw_ratings = load_ratings(config_path=config_path, verbose=cfg.get("verbose", False))
    mid_map = {m: i for i, m in enumerate(raw_ratings.MovieID.unique())}

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
        )  # both (1, d_emb)

        # 2) item tables
        V_gmf   = model.item_gmf.weight      # (N, d_emb)
        V_mlp   = model.item_mlp.weight      # (N, d_emb)
        V_demog = model.item_demog.weight    # (N, demog_feat_dim)

        # 3) GMF branch
        gmf_vec = u_gmf * V_gmf               # broadcasts to (N, d_emb)
        s_gmf   = model.out_gmf(gmf_vec).squeeze(1)

        # 4) MLP branch
        um        = u_mlp.repeat(len(V_mlp), 1)    # (N, d_emb)
        mlp_in    = torch.cat([um, V_mlp], dim=1)  # (N, 2*d_emb)
        mlp_feats = model.mlp_head(mlp_in)        # (N, mlp_last)
        s_mlp     = model.out_mlp(mlp_feats).squeeze(1)

        # 5) Demographic branch
        g = model.gender_emb(demog_row[[0]])
        raw_age = demog_row[1].item()
        age_idx = AGE2IDX.get(raw_age, 0)
        a = model.age_emb(torch.tensor([age_idx], device=demog_row.device))
        o = model.occ_emb(demog_row[2].unsqueeze(0))
        z = model.zip_emb(demog_row[3].unsqueeze(0))
        dem_feat = torch.cat([g, a, o, z], dim=1)    # (1, demog_feat_dim)
        dem_rep  = dem_feat.repeat(len(V_demog), 1)  # (N, demog_feat_dim)
        dem_in   = dem_rep * V_demog                 # (N, demog_feat_dim)
        s_demog  = model.out_demog(dem_in).squeeze(1)

        # 6) normalize for gate
        gmf_n   = model.norm_gmf(gmf_vec)
        mlp_n   = model.norm_mlp(mlp_feats)
        dem_n   = model.norm_demog(dem_in)

        # 7) gated fusion
        gate_in  = torch.cat([gmf_n, mlp_n, dem_n], dim=1)
        logits   = model.gate_linear(gate_in)
        wts      = F.softmax(logits / model.temperature, dim=1)
        fused    = wts[:,0]*s_gmf + wts[:,1]*s_mlp + wts[:,2]*s_demog

        return fused

if mode == "Existing User":
    uid = st.selectbox("Select User ID", sorted(ratings_df.uid.unique()))
    d = demog_tensor[uid]
    st.markdown(f"**Gender:** {'F' if d[0]==1 else 'M'}  ")
    st.markdown(f"**Age bucket:** {int(d[1])}  ")
    st.markdown(f"**Occupation code:** {int(d[2])}  ")
    st.markdown(f"**ZIP prefix:** {int(d[3])}  ")

    user_hist = ratings_df[ratings_df.uid==uid].sort_values("Timestamp", ascending=False)
    max_k = min(len(user_hist), 10)
    k = st.slider("Support-set size (k)", 1, max_k, 5)
    support = user_hist.head(k)

    st.markdown("#### Support Ratings (edit if you like)")
    support_df = pd.DataFrame({
        "Movie":  [titles[mid] for mid in support.mid],
        "Rating": support.Rating.values
    })
    edited = st.data_editor(support_df, num_rows="fixed", key="support_editor")

    supp_m = torch.tensor(
        [titles.index(m) for m in edited["Movie"]],
        dtype=torch.long
    )
    supp_r = torch.tensor(
        edited["Rating"].astype(float).tolist(),
        dtype=torch.float
    )

    if st.button("Recommend"):
        scores   = vectorized_scores(model, d, supp_m, supp_r)
        residuals = scores - global_means
        counts = ratings_df.groupby("mid").size()
        popular = counts[counts >= 20].index  # only mids with ≥20 ratings
        residuals[~torch.tensor([m in popular for m in range(len(residuals))])] = -1e6
        vals, idxs = residuals.topk(topk)

        st.markdown("### Top Recommendations")
        for delta, idx in zip(vals.tolist(), idxs.tolist()):
            raw = scores[idx].item()
            st.write(f"- **{titles[idx]}** (pred {raw:.2f}, Δ={delta:+.2f})")

else:  # New User
    st.markdown("### Enter New User Profile")

    age_key = st.selectbox("Age bucket", options=list(AGE_BUCKETS.keys()),
                           format_func=lambda x: AGE_BUCKETS[x])
    gender  = st.radio("Gender", ["M", "F"])
    occ_key = st.selectbox("Occupation", options=list(OCCUPATIONS.keys()),
                           format_func=lambda x: OCCUPATIONS[x])
    zp      = st.text_input("ZIP prefix (first 2 digits)", "63")

    d_row = torch.tensor([
        1 if gender=="F" else 0,
        age_key,
        occ_key,
        int(zp)
    ], dtype=torch.long)

    k = st.sidebar.slider("Number of support ratings (k)", 0, 10, 0)
    supp_m_list, supp_r_list = [], []
    if k > 0:
        st.markdown(f"#### Enter {k} sample ratings")
        for i in range(k):
            movie  = st.selectbox(f"Movie #{i+1}", titles, key=f"new_m{i}")
            rating = st.slider(f"Rating for '{movie}'", 1.0, 5.0, 3.0, key=f"new_r{i}")
            supp_m_list.append(titles.index(movie))
            supp_r_list.append(rating)

    supp_m = torch.tensor(supp_m_list, dtype=torch.long)
    supp_r = torch.tensor(supp_r_list, dtype=torch.float)

    if st.button("Recommend"):
        scores   = vectorized_scores(model, d_row, supp_m, supp_r)
        residuals = scores - global_means
        counts = ratings_df.groupby("mid").size()
        popular = counts[counts >= 20].index  # only mids with ≥20 ratings
        residuals[~torch.tensor([m in popular for m in range(len(residuals))])] = -1e6
        vals, idxs = residuals.topk(topk)

        st.markdown("### Top Recommendations")
        for delta, idx in zip(vals.tolist(), idxs.tolist()):
            raw = scores[idx].item()
            st.write(f"- **{titles[idx]}** (pred {raw:.2f}, Δ={delta:+.2f})")