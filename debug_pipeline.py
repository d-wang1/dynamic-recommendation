import torch
from pathlib import Path
from data.objs import AGE2IDX

from data.helper import (
    load_config,
    gen_demographic_table,
    load_save_neumf_table,
    load_movies,
    load_ratings,
)
from models.hyper_neumf import DCHyperNeuMF

def vectorized_scores(model, demog_row, supp_m, supp_r):
    """
    Score every movie for one user in one go using the vector‐GMF + MLP + demog fusion.
    Returns a tensor of shape (n_items,) of raw predictions.
    """
    with torch.no_grad():
        # 1) Build user embeddings
        u_gmf, u_mlp = model.make_user_embs(
            demog_row.unsqueeze(0),            # (1, demog_dim)
            supp_m.unsqueeze(0),               # (1, k)
            supp_r.unsqueeze(0),               # (1, k)
        )                                       # each (1, d_emb)

        # 2) Frozen item tables
        V_gmf   = model.item_gmf.weight       # (n_items, d_emb)
        V_mlp   = model.item_mlp.weight       # (n_items, d_emb)
        V_demog = model.item_demog.weight     # (n_items, demog_feat_dim)

        # 3) GMF branch
        gmf_vec = u_gmf * V_gmf                # (n_items, d_emb)
        s_gmf   = model.out_gmf(gmf_vec).squeeze(1)

        # 4) MLP branch
        um        = u_mlp.repeat(len(V_mlp), 1)
        mlp_feat  = model.mlp_head(torch.cat([um, V_mlp], dim=1))
        s_mlp     = model.out_mlp(mlp_feat).squeeze(1)

        # 5) Demog branch
        # rebuild demog_feats exactly as in forward:
        # (you can also call model._build_demog_feats(demog_row) if you factor that out)
        g = model.gender_emb(demog_row[[0]])
        raw_age   = demog_row[1].item()                       # e.g. 18 or 56
        idx0_6    = AGE2IDX.get(raw_age, 0)                   # map into 0..6
        age_tensor= torch.tensor([idx0_6], device=demog_row.device)
        a         = model.age_emb(age_tensor)
        o = model.occ_emb(demog_row[2].unsqueeze(0))
        z = model.zip_emb(demog_row[3].unsqueeze(0))
        dem_feats = torch.cat([g, a, o, z], dim=1).repeat(len(V_demog), 1)
        dem_in    = dem_feats * V_demog
        s_demog   = model.out_demog(dem_in).squeeze(1)

        # 6) Fuse via gating
        # normalize
        n_g = model.norm_gmf(gmf_vec)
        n_m = model.norm_mlp(mlp_feat)
        n_d = model.norm_demog(dem_in)
        # k-feature (optional; remove if you dropped it)
        k = supp_m.numel()
        max_k = getattr(model.hparams, "max_k", k)
        k_feat = torch.full((len(V_gmf),1), fill_value=k/float(max_k), device=gmf_vec.device)
        # build gate input
        gate_in = torch.cat([n_g, n_m, n_d, k_feat], dim=1)
        logits  = model.gate_linear(gate_in)
        wts     = torch.softmax(logits / model.temperature, dim=1)

        # final
        fused = wts[:,0]*s_gmf + wts[:,1]*s_mlp + wts[:,2]*s_demog
        return fused

def main():
    # ─── 1) Setup ─────────────────────────────────────────────────────────
    cfg = load_config("config.json")

    # pretrained NeuMF tables
    item_gmf, item_mlp = load_save_neumf_table(
        config_path="config.json",
        verbose=cfg.get("verbose", False),
    )

    # rebuild & freeze your HyperNeuMF
    ckpt_path = Path(cfg["app"]["ckpt_to_use"]).expanduser().resolve()
    model: DCHyperNeuMF = DCHyperNeuMF.load_from_checkpoint(str(ckpt_path))
    model.item_gmf.weight.data.copy_(item_gmf)
    model.item_mlp.weight.data.copy_(item_mlp)
    for p in model.item_gmf.parameters():
        p.requires_grad_(False)
    for p in model.item_mlp.parameters():
        p.requires_grad_(False)
    model.eval()

    # demographics & ratings
    ratings_df, demog_tensor, n_users, n_items = gen_demographic_table(
        config_path="config.json",
        verbose=cfg.get("verbose", False),
    )
    print(f"Loaded {n_users} users, {n_items} movies, {len(ratings_df)} ratings.")

    # movie titles
    raw_ratings = load_ratings("config.json")
    mid_map = {m:i for i,m in enumerate(raw_ratings.MovieID.unique())}
    movies_df = load_movies("config.json")
    titles = [None]*n_items
    for _, row in movies_df.iterrows():
        idx = mid_map.get(row.MovieID)
        if idx is not None:
            titles[idx] = row.Title

    # global mean popularity
    gm = ratings_df.groupby("mid")["Rating"].mean().reset_index()
    global_means = torch.zeros(n_items)
    for _, r in gm.iterrows():
        global_means[int(r.mid)] = r.Rating


    print(ratings_df.head())
    # ─── 2) Choose a user & support-set ────────────────────────────────────
    uid = 1
    user_hist = ratings_df[ratings_df.uid==uid].sort_values("Timestamp", ascending=False)
    k = 3
    support = user_hist.head(k)
    supp_m = torch.tensor(support.mid.values, dtype=torch.long)
    supp_r = torch.tensor(support.Rating.values, dtype=torch.float)
    demog_row = demog_tensor[uid]

    # ─── 3) Score & display ────────────────────────────────────────────────
    USE_DELTAS = True    # False to sort by raw score
    USE_SHRINK = True    # apply Bayesian shrinkage toward overall mean
    USE_FILTER = False   # filter out ultra‐rare movies (< min_count)

    # 1) get your model’s raw scores for all n_items
    scores = vectorized_scores(model, demog_row, supp_m, supp_r)  # (n_items,)
    device = scores.device
    n_items = scores.size(0)

    if USE_DELTAS:
        # 2) build counts as a torch tensor on the same device
        counts_series = ratings_df.groupby("mid").size().reindex(range(n_items), fill_value=0)
        counts = torch.tensor(counts_series.values,
                            dtype=torch.float32,
                            device=device)          # (n_items,)

        # 3) your global_means is already a torch tensor of shape (n_items,)
        raw_means = global_means.to(device)  # just ensure it’s on the right device

        # 4) overall float mean of all ratings
        overall_mean = float(ratings_df.Rating.mean())

        if USE_SHRINK:
            m = 20.0
            # Bayesian shrinkage: (count * item_mean + m * global_mean) / (count + m)
            pop_baseline = (counts * raw_means + m * overall_mean) / (counts + m)
        else:
            pop_baseline = raw_means

        # 5) compute deltas
        deltas = scores - pop_baseline

        # 6) optionally filter out ultra‐rare movies
        if USE_FILTER:
            min_count = 20
            mask = counts >= min_count                            # bool tensor
            deltas = torch.where(mask, deltas, torch.full_like(deltas, -1e6))

        # 7) take Top-K by Δ
        top_vals, top_idxs = deltas.topk(10)

        print("\nTop-10 by Δ:")
        for delta, idx in zip(top_vals.tolist(), top_idxs.tolist()):
            print(f"- {titles[idx]}  "
                f"(pred={scores[idx]:.2f}, pop={pop_baseline[idx]:.2f}, Δ={delta:+.2f})")

    else:
        # Top-K by raw prediction
        top_vals, top_idxs = scores.topk(10)
        print("\nTop-10 by raw score:")
        for pred, idx in zip(top_vals.tolist(), top_idxs.tolist()):
            pop = global_means[idx].item()
            print(f"- {titles[idx]}  (pred={pred:.2f}, pop={pop:.2f}, Δ={pred - pop:+.2f})")

if __name__ == "__main__":
    main()