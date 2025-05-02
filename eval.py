import torch
from pathlib import Path
from torch.utils.data import DataLoader

from data.helper import (
    load_config,
    gen_datamodule,
    gen_demographic_table,      # ← new
    load_save_neumf_table,
    load_movies,
    load_ratings,
)
from models.hyper_neumf import DCHyperNeuMF
from data.objs import AGE2IDX

def vectorized_scores(model, demog_row, supp_m, supp_r):
    with torch.no_grad():
        # 1) user embeddings
        u_gmf, u_mlp = model.make_user_embs(
            demog_row.unsqueeze(0),
            supp_m.unsqueeze(0),
            supp_r.unsqueeze(0),
        )
        # 2) frozen tables
        V_gmf   = model.item_gmf.weight
        V_mlp   = model.item_mlp.weight
        V_demog = model.item_demog.weight

        # 3) GMF branch
        gmf_vec = u_gmf * V_gmf
        s_gmf   = model.out_gmf(gmf_vec).squeeze(1)

        # 4) MLP branch
        um       = u_mlp.repeat(len(V_mlp), 1)
        mlp_feat = model.mlp_head(torch.cat([um, V_mlp], dim=1))
        s_mlp    = model.out_mlp(mlp_feat).squeeze(1)

        # 5) Demog branch
        g = model.gender_emb(demog_row[[0]])
        raw_age = demog_row[1].item()
        age_idx   = AGE2IDX.get(raw_age, 0)   # default to bucket 0 if unseen
        age_tensor = torch.tensor([age_idx], device=demog_row.device, dtype=torch.long)
        a = model.age_emb(age_tensor)
        o = model.occ_emb(demog_row[2].unsqueeze(0))
        z = model.zip_emb(demog_row[3].unsqueeze(0))
        dem_feats = torch.cat([g, a, o, z], dim=1).repeat(len(V_demog), 1)
        dem_in    = dem_feats * V_demog
        s_demog   = model.out_demog(dem_in).squeeze(1)

        # 6) gating + fuse
        n_g = model.norm_gmf(gmf_vec)
        n_m = model.norm_mlp(mlp_feat)
        n_d = model.norm_demog(dem_in)
        k   = supp_m.numel()
        max_k = model.hparams.max_k
        k_feat = torch.full((len(V_gmf),1), k/float(max_k), device=gmf_vec.device)
        gate_in = torch.cat([n_g, n_m, n_d, k_feat], dim=1)
        logits  = model.gate_linear(gate_in)
        wts     = torch.softmax(logits / model.temperature, dim=1)

        return wts[:,0]*s_gmf + wts[:,1]*s_mlp + wts[:,2]*s_demog

def compute_rmse(model, datamodule, device="cpu"):
    loader = datamodule.val_dataloader()
    model.to(device).eval()
    total_se, total_n = 0.0, 0
    with torch.no_grad():
        for demog, s_m, s_r, q_m, q_r in loader:
            demog, s_m, s_r, q_m, q_r = [t.to(device) for t in (demog, s_m, s_r, q_m, q_r)]
            preds, _ = model(demog, s_m, s_r, q_m)
            total_se += (preds - q_r).pow(2).sum().item()
            total_n += q_r.numel()
    return (total_se/total_n)**0.5

def compute_precision_at_k(
    model,
    datamodule,
    titles,
    global_means,
    ratings_df,            # ← now required
    K=5,
    method="raw",          # "raw", "residual", or "popularity"
    shrinkage=False,
    device="cpu"
):
    loader = datamodule.val_dataloader()
    model.to(device).eval()

    # build counts & pop_baseline from ratings_df + global_means
    n_items = len(global_means)
    counts_series = ratings_df.groupby("mid").size().reindex(range(n_items), fill_value=0)
    counts = torch.tensor(counts_series.values, dtype=torch.float, device=device)
    raw_means = global_means.to(device)
    overall_mean = float(ratings_df.Rating.mean())

    if method == "residual":
        if shrinkage:
            m = 20.0
            pop_baseline = (counts * raw_means + m * overall_mean) / (counts + m)
        else:
            pop_baseline = raw_means

    hits, total = 0, 0
    with torch.no_grad():
        for demog, s_m, s_r, q_m, q_r in loader:
            demog, s_m, s_r = demog.to(device), s_m.to(device), s_r.to(device)
            for i in range(demog.size(0)):
                dr       = demog[i]
                sm, sr   = s_m[i], s_r[i]
                true_mid = q_m[i].item()

                scores = vectorized_scores(model, dr, sm, sr)
                if method == "residual":
                    scores = scores - pop_baseline
                elif method == "popularity":
                    scores = counts

                topk = torch.topk(scores, K).indices.tolist()
                if true_mid in topk:
                    hits += 1
                total += 1

    return hits / total if total > 0 else 0.0

def main():
    cfg = load_config("config.json")

    # 1) load full ratings + demogs
    ratings_df, demog_tensor, n_users, n_items = gen_demographic_table(
        config_path="config.json",
        verbose=cfg.get("verbose", False),
    )

    # 2) build datamodule (uses same underlying ratings_df internally)
    dm, _, _, _ = gen_datamodule(
        k_list    = cfg["model_hyperparams"]["k"],
        max_k     = cfg["model_hyperparams"]["max_k"],
        test_size = 0.2,
        random_state=42,
        verbose   = cfg.get("verbose", False),
    )

    # 3) pretrained NeuMF tables
    item_gmf, item_mlp = load_save_neumf_table("config.json")

    # 4) load & freeze HyperNeuMF
    ckpt  = Path(cfg["app"]["ckpt_to_use"]).expanduser().resolve()
    model = DCHyperNeuMF.load_from_checkpoint(str(ckpt))
    model.item_gmf.weight.data.copy_(item_gmf)
    model.item_mlp.weight.data.copy_(item_mlp)
    for p in model.item_gmf.parameters():
        p.requires_grad_(False)
    for p in model.item_mlp.parameters():
        p.requires_grad_(False)
    model.eval()

    # 5) titles
    raw_ratings = load_ratings("config.json")
    mid_map = {m: i for i, m in enumerate(raw_ratings.MovieID.unique())}
    movies_df = load_movies("config.json")
    titles = [None] * n_items
    for _, row in movies_df.iterrows():
        idx = mid_map.get(row.MovieID)
        if idx is not None:
            titles[idx] = row.Title

    # 6) global_means from ratings_df
    gm = ratings_df.groupby("mid")["Rating"].mean().reset_index()
    global_means = torch.zeros(n_items)
    for _, r in gm.iterrows():
        global_means[int(r.mid)] = r.Rating

    # 7) RMSE
    rmse = compute_rmse(model, dm, device="cpu")
    print(f"\nValidation  RMSE: {rmse:.4f}\n")

    # 8) Precision@5 under three scoring options
    for method in ["raw", "residual", "popularity"]:
        prec = compute_precision_at_k(
            model, dm, titles, global_means,
            ratings_df=ratings_df,   # ← pass in your full ratings
            K=5, method=method,
            shrinkage=(method=="residual"),
            device="cpu",
        )
        print(f"Precision@5 ({method}): {prec:.4f}")

if __name__ == "__main__":
    main()