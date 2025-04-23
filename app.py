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

def vectorized_scores(model, demog_row, supp_m, supp_r, device=None):
    """
    Compute model.forward on *every* movie in one giant batch,
    so the vectorized version is 100% the same as model.forward().
    Returns:
      preds: (n_items,) float tensor
      gates: (n_items, 3) float tensor  (if your forward returns (pred, wts))
    """
    model.eval()
    # what device to live on?
    if device is None:
        device = model.item_gmf.weight.device

    # 1) make a 0..N-1 tensor of every movie ID
    N = model.item_gmf.num_embeddings
    all_mids = torch.arange(N, device=device, dtype=torch.long)

    # 2) replicate the single‐user info N times
    #    demog_row: (demog_dim,) → (N, demog_dim)
    d_rep = demog_row.unsqueeze(0).repeat(N, 1)
    #    supp_m: (k,) → (N, k), same for supp_r
    s_m_rep = supp_m.unsqueeze(0).repeat(N, 1)
    s_r_rep = supp_r.unsqueeze(0).repeat(N, 1)

    # 3) call your model
    with torch.no_grad():
        out = model(d_rep, s_m_rep, s_r_rep, all_mids)
        # your forward may return either `preds` or `(preds, gates)`
        if isinstance(out, tuple):
            preds, gates = out
        else:
            preds, gates = out, None

    # 4) squeeze off the extra dim if needed
    preds = preds.view(-1)
    return preds, gates

if mode == "Existing User":
    uid = st.selectbox("Select User ID", sorted(ratings_df.uid.unique()))
    d = demog_tensor[uid]
    st.markdown(f"**Gender:** {'F' if d[0]==1 else 'M'}  ")
    st.markdown(f"**Age bucket:** {int(d[1])}  ")
    st.markdown(f"**Occupation code:** {int(d[2])}  ")
    st.markdown(f"**ZIP prefix:** {int(d[3])}  ")

    # user_hist = ratings_df[ratings_df.uid==uid].sort_values("Timestamp", ascending=False)
    # max_k = min(len(user_hist), 10)
    # k = st.slider("Support-set size (k)", 1, max_k, 5)
    # support = user_hist.head(k)

    #DEBUG: support ratings
    user_hist = ratings_df[ratings_df.uid==uid] \
                .sort_values("Timestamp", ascending=False)
    support   = user_hist.head(5)

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
        raw_direct, gates_app = model(
        d.unsqueeze(0),
        supp_m.unsqueeze(0),
        supp_r.unsqueeze(0),
        torch.tensor([1311])
        )
        st.write("APP direct → ", raw_direct.item(), "gates→", gates_app.tolist())
        scores,_   = vectorized_scores(model, d, supp_m, supp_r)
        residuals = scores - global_means
        counts = ratings_df.groupby("mid").size()
        popular = counts[counts >= 5].index  # only mids with ≥20 ratings
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
        scores,_   = vectorized_scores(model, d_row, supp_m, supp_r)
        residuals = scores - global_means
        counts = ratings_df.groupby("mid").size()
        popular = counts[counts >= 5].index  # only mids with ≥20 ratings
        residuals[~torch.tensor([m in popular for m in range(len(residuals))])] = -1e6
        vals, idxs = residuals.topk(topk)

        st.markdown("### Top Recommendations")
        for delta, idx in zip(vals.tolist(), idxs.tolist()):
            raw = scores[idx].item()
            st.write(f"- **{titles[idx]}** (pred {raw:.2f}, Δ={delta:+.2f})")