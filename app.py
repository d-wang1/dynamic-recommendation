import asyncio
import os
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd

from data.helper import (
    load_config,
    load_movies,
    load_ratings,
    gen_demographic_table,
    load_save_neumf_table,
)
from models.hyper_neumf import DCHyperNeuMF
from data.objs import AGE2IDX, AGE_BUCKETS, OCCUPATIONS

# ─── 1) Boilerplate to silence streamlit/tqdm/Torch classes bug ───────────────
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"]      = "false"


# ─── 2) Load & cache everything once ──────────────────────────────────────────
@st.cache_resource
def setup_everything(config_path: str = "config.json"):
    # 2a) config
    cfg = load_config(config_path=config_path, verbose= False)

    # 2b) extract & freeze vanilla NeuMF tables
    item_gmf, item_mlp = load_save_neumf_table(
        config_path=config_path, verbose=cfg.get("verbose", False)
    )

    # 2c) rebuild your HyperNeuMF from checkpoint
    ckpt = Path(cfg["app"]["ckpt_to_use"]).expanduser().resolve()
    model: DCHyperNeuMF = DCHyperNeuMF.load_from_checkpoint(str(ckpt))
    model.item_gmf.weight.data.copy_(item_gmf)
    model.item_mlp.weight.data.copy_(item_mlp)
    for p in model.item_gmf.parameters():
        p.requires_grad_(False)
    for p in model.item_mlp.parameters():
        p.requires_grad_(False)
    model.eval()

    # 2d) load ratings + demographics
    ratings_df, demog_tensor, n_users, n_items = gen_demographic_table(
        config_path=config_path, verbose=cfg.get("verbose", False)
    )

    # 2e) raw→internal ID map
    raw_ratings = load_ratings(config_path=config_path, verbose=cfg.get("verbose", False))
    mid_map = {int(m): i for i, m in enumerate(raw_ratings.MovieID.unique())}

    # 2f) load titles, skip any unrated movie
    movies_df = load_movies(config_path=config_path, verbose=cfg.get("verbose", False))
    titles = [None] * n_items
    for _, row in movies_df.iterrows():
        internal = mid_map.get(int(row.MovieID))
        if internal is not None:
            titles[internal] = row.Title

    # 2g) precompute global means for residual ranking
    gm = ratings_df.groupby("mid")["Rating"].mean().reset_index()
    global_means = torch.zeros(n_items)
    for _, r in gm.iterrows():
        global_means[int(r.mid)] = r.Rating

    return model, ratings_df, demog_tensor, titles, global_means


model, ratings_df, demog_tensor, titles, global_means = setup_everything()


# ─── 3) Streamlit UI ──────────────────────────────────────────────────────────
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
        )  # (1, d_emb)

        # 2) frozen item tables
        V_gmf = model.item_gmf.weight      # (N, d_emb)
        V_mlp = model.item_mlp.weight      # (N, d_emb)
        V_dem = model.item_demog.weight    # (N, demog_feat_dim)

        # 3) branch vectors
        gmf_vec  = u_gmf * V_gmf            # (N, d_emb)
        um       = u_mlp.repeat(len(V_mlp), 1)
        mlp_vec  = model.mlp_head(torch.cat([um, V_mlp], dim=1))  # (N, mlp_last)

        # 4) demog branch vector
        g = model.gender_emb(demog_row[[0]])
        age_idx = AGE2IDX[int(demog_row[1].item())]
        a = model.age_emb(torch.tensor([age_idx], device=demog_row.device))
        o = model.occ_emb(demog_row[2].unsqueeze(0))
        z = model.zip_emb(demog_row[3].unsqueeze(0))
        dem_feat = torch.cat([g, a, o, z], dim=1).repeat(len(V_dem), 1)
        dem_vec  = dem_feat * V_dem

        # 5) branch scores
        s_gmf = model.out_gmf(gmf_vec).squeeze(1)
        s_mlp = model.out_mlp(mlp_vec).squeeze(1)
        s_dem = model.out_demog(dem_vec).squeeze(1)

        # 6) normalize for gating
        n_g = model.norm_gmf(gmf_vec)
        n_m = model.norm_mlp(mlp_vec)
        n_d = model.norm_demog(dem_vec)

        # 7) *** k‐feature ***
        k = supp_m.numel()                # how many support ratings
        max_k = model.hparams.max_k      # saved from __init__
        k_feat = torch.full((len(V_gmf), 1),
                            fill_value=k/float(max_k),
                            device=gmf_vec.device)

        # 8) build gate input of size (=56+1=57)
        gate_in = torch.cat([n_g, n_m, n_d, k_feat], dim=1)  # (N,57)
        logits  = model.gate_linear(gate_in)                 # (N,3)
        wts     = F.softmax(logits / model.temperature, dim=1)

        # 9) fuse
        fused   = wts[:,0]*s_gmf + wts[:,1]*s_mlp + wts[:,2]*s_dem
        return fused

if mode == "Existing User":
    # — pick an existing user and show demographics —
    uid = st.selectbox("Select User ID", sorted(ratings_df.uid.unique()))
    d = demog_tensor[uid]
    st.markdown(f"**Gender:** {'F' if d[0]==1 else 'M'}  ")
    st.markdown(f"**Age bucket:** {int(d[1])}  ")
    st.markdown(f"**Occupation code:** {int(d[2])}  ")
    st.markdown(f"**ZIP prefix:** {int(d[3])}  ")

    # — build or edit support set —
    hist = ratings_df[ratings_df.uid==uid].sort_values("Timestamp", ascending=False)
    max_k = min(len(hist), 10)
    k = st.slider("Support-set size (k)", 1, max_k, 5)
    support = hist.head(k)
    df_sup = pd.DataFrame({
        "Movie": [titles[mid] for mid in support.mid],
        "Rating": support.Rating.values
    })
    edited = st.data_editor(df_sup, num_rows="fixed", key="sup_ed")

    supp_m = torch.tensor([titles.index(m) for m in edited["Movie"]], dtype=torch.long)
    supp_r = torch.tensor(edited["Rating"].astype(float).tolist(), dtype=torch.float)

    if st.button("Recommend"):
        scores   = vectorized_scores(model, d, supp_m, supp_r)
        residual = scores - global_means
        vals, idxs = residual.topk(topk)

        st.markdown("### Top Recommendations (by residual)")
        for Δ, idx in zip(vals.tolist(), idxs.tolist()):
            raw = scores[idx].item()
            st.write(f"- **{titles[idx]}**  (pred {raw:.2f}, Δ={Δ:+.2f})")

else:
    # — new user form —
    gender = st.radio("Gender", ["M","F"])
    age_key= st.selectbox("Age bucket", list(AGE_BUCKETS.keys()),
                          format_func=lambda x: AGE_BUCKETS[x])
    occ_key= st.selectbox("Occupation", list(OCCUPATIONS.keys()),
                          format_func=lambda x: OCCUPATIONS[x])
    zp     = st.text_input("ZIP prefix (2 digits)", "63")

    drow = torch.tensor([
        1 if gender=="F" else 0,
        age_key,
        occ_key,
        int(zp)
    ], dtype=torch.long)

    k = st.slider("Number of support ratings (k)", 0, 10, 0)
    supp_m, supp_r = [], []
    if k>0:
        st.markdown(f"#### Enter {k} sample ratings")
        for i in range(k):
            mv = st.selectbox(f"Movie #{i+1}", titles, key=f"nm{i}")
            rt = st.slider(f"Rating for '{mv}'", 1.0,5.0,3.0, key=f"nr{i}")
            supp_m.append(titles.index(mv))
            supp_r.append(rt)
    supp_m = torch.tensor(supp_m, dtype=torch.long)
    supp_r = torch.tensor(supp_r, dtype=torch.float)

    if st.button("Recommend"):
        scores   = vectorized_scores(model, drow, supp_m, supp_r)
        residual = scores - global_means
        vals, idxs = residual.topk(topk)

        st.markdown("### Top Recommendations (by residual)")
        for Δ, idx in zip(vals.tolist(), idxs.tolist()):
            raw = scores[idx].item()
            st.write(f"- **{titles[idx]}**  (pred {raw:.2f}, Δ={Δ:+.2f})")