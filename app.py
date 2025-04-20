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

    return model, ratings_int, demog_tensor, titles

# Initialize everything
model, ratings_df, demog_tensor, titles = setup_everything("config.json")


# --- 2. Streamlit GUI ----------------------------------------------------
st.title("🎬 Cold-Start Recommender Demo")

mode = st.sidebar.radio("Mode", ["Existing User", "New User"])
topk = st.sidebar.slider("Top-K", 5, 20, 10)


def vectorized_scores(model, demog_row, supp_m, supp_r):
    """
    Scores all movies in one go with the vector‐GMF branch.
    Returns a tensor of shape (n_items,) of raw scores.
    """
    with torch.no_grad():
        # 1) Make user vectors (1, d_emb)
        u_gmf, u_mlp = model.make_user_embs(
            demog_row.unsqueeze(0),
            supp_m.unsqueeze(0),
            supp_r.unsqueeze(0),
        )
        # 2) Pull frozen tables
        V_gmf = model.item_gmf.weight    # (n_items, d_emb)
        V_mlp = model.item_mlp.weight    # (n_items, d_emb)

        # 3) VECTOR GMF branch
        gmf = u_gmf * V_gmf              # (n_items, d_emb)

        # 4) MLP branch
        um = u_mlp.repeat(len(V_mlp), 1)                     # (n_items, d_emb)
        mlp_feats = model.mlp_head(torch.cat([um, V_mlp], 1))  # (n_items, mlp_dim)

        # 5) Fuse & score
        feats  = torch.cat([gmf, mlp_feats], dim=1)  # (n_items, d_emb+mlp_dim)
        scores = model.out(feats).squeeze(1)         # (n_items,)
        return scores


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

        st.markdown("### Top Recommendations")
        for score, idx in zip(vals.cpu().tolist(), idxs.cpu().tolist()):
            # print(score, idx)
            st.write(f"- **{titles[idx]}** (predicted {score:.2f})")

else:  # New User
    st.markdown("### Enter New User Profile")
    gender = st.radio("Gender", ["M", "F"])
    age    = st.selectbox("Age bucket", list(range(1, 8)))
    occ    = st.number_input("Occupation code (0–20)", 0, 20, 0)
    zp     = st.text_input("ZIP prefix (2 digits)", "63")
    d_row  = torch.tensor(
        [1 if gender=="F" else 0, age, occ, int(zp)],
        dtype=torch.long
    )

    k = st.slider("Number of support ratings (k)", 0, 10, 0)
    supp_m_list, supp_r_list = [], []
    if k > 0:
        st.markdown(f"#### Enter {k} sample ratings")
        for i in range(k):
            movie = st.selectbox(f"Movie #{i+1}", titles, key=f"m{i}")
            rating= st.slider(f"Rating for '{movie}'", 1.0, 5.0, 3.0, key=f"r{i}")
            supp_m_list.append(titles.index(movie))
            supp_r_list.append(rating)
    # build the tensors—even if empty
    supp_m = torch.tensor(supp_m_list, dtype=torch.long)
    supp_r = torch.tensor(supp_r_list, dtype=torch.float)

    if st.button("Recommend"):
        # st.write("Support movie IDs:", supp_m.tolist())
        # st.write("Support ratings:  ", supp_r.tolist())
        # u_gmf, u_mlp = model.make_user_embs(d_row.unsqueeze(0), supp_m.unsqueeze(0), supp_r.unsqueeze(0))
        # st.write("u_gmf[:5]:", u_gmf[0,:5].tolist())
        scores = vectorized_scores(model, d_row, supp_m, supp_r)  # (n_items,)
        vals, idxs = scores.topk(topk)
        st.markdown("### Top Recommendations")
        for score, idx in zip(vals.cpu().tolist(), idxs.cpu().tolist()):
            # print(score, idx)
            st.write(f"- **{titles[idx]}** (predicted {score:.2f})")