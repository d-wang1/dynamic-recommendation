# eval.py

import torch
from pathlib import Path
from torch.utils.data import DataLoader

from data.helper import (
    load_config,
    gen_demographic_table,
    load_save_neumf_table,
    gen_datamodule,
    load_movies,
    load_ratings,
)
from models.hyper_neumf import DCHyperNeuMF
from data.objs import AGE2IDX

# ─── unchanged ────────────────────────────────────────────────────────────
def vectorized_scores(model, demog_row, supp_m, supp_r):
    with torch.no_grad():
        u_gmf, u_mlp = model.make_user_embs(
            demog_row.unsqueeze(0),
            supp_m.unsqueeze(0),
            supp_r.unsqueeze(0),
        )
        V_gmf   = model.item_gmf.weight
        V_mlp   = model.item_mlp.weight
        V_demog = model.item_demog.weight

        # GMF branch
        gmf_vec = u_gmf * V_gmf
        s_gmf   = model.out_gmf(gmf_vec).squeeze(1)

        # MLP branch
        um       = u_mlp.repeat(len(V_mlp),1)
        mlp_feat = model.mlp_head(torch.cat([um, V_mlp], dim=1))
        s_mlp    = model.out_mlp(mlp_feat).squeeze(1)

        # Demog branch
        g = model.gender_emb(demog_row[[0]])
        raw_age = demog_row[1].item()
        idx0_6  = AGE2IDX.get(raw_age, 0)
        a = model.age_emb(torch.tensor([idx0_6], device=demog_row.device))
        o = model.occ_emb(demog_row[2].unsqueeze(0))
        z = model.zip_emb(demog_row[3].unsqueeze(0))
        dem_feats = torch.cat([g,a,o,z],dim=1).repeat(len(V_demog),1)
        dem_in    = dem_feats * V_demog
        s_demog   = model.out_demog(dem_in).squeeze(1)

        # Gate & fuse
        n_g  = model.norm_gmf(gmf_vec)
        n_m  = model.norm_mlp(mlp_feat)
        n_d  = model.norm_demog(dem_in)
        k    = supp_m.numel()
        maxk = model.hparams.max_k
        kf   = torch.full((len(V_gmf),1), k/float(maxk), device=gmf_vec.device)
        gin  = torch.cat([n_g,n_m,n_d,kf], dim=1)
        logits= model.gate_linear(gin)
        wts  = torch.softmax(logits/model.temperature, dim=1)

        return wts[:,0]*s_gmf + wts[:,1]*s_mlp + wts[:,2]*s_demog


# ─── 1) RMSE (unchanged) ─────────────────────────────────────────────────
def compute_rmse(model, datamodule, device="cpu"):
    loader = datamodule.val_dataloader()
    model.to(device).eval()
    total_se, total_n = 0.0, 0
    with torch.no_grad():
        for d,s_m,s_r,q_m,q_r in loader:
            d,s_m,s_r,q_m,q_r = [t.to(device) for t in (d,s_m,s_r,q_m,q_r)]
            preds,_ = model(d,s_m,s_r,q_m)
            total_se += (preds-q_r).pow(2).sum().item()
            total_n  += q_r.numel()
    return (total_se/total_n)**0.5


# ─── 2) Precision@5 with multiple ranking methods ────────────────────────
def compute_precision_at_5(
    model,
    datamodule,
    counts: torch.Tensor,
    raw_means: torch.Tensor,
    stddevs: torch.Tensor,
    overall_mean: float,
    K=5,
    method="raw",        # "raw", "residual", "popularity", "zscore"
    shrinkage=True,      # only used if method=="residual"
    filter_count:int=10, # or None to disable
    device="cpu"
):
    loader = datamodule.val_dataloader()
    model.to(device).eval()

    # precompute popularity baseline for "residual"
    if method=="residual":
        m = 20.0
        pop_baseline = (counts * raw_means + m*overall_mean) / (counts + m)
        pop_baseline = pop_baseline.to(device)

    hits, total = 0, 0
    with torch.no_grad():
        for d, s_m, s_r, q_m, _ in loader:
            d, s_m, s_r = d.to(device), s_m.to(device), s_r.to(device)
            for i in range(d.size(0)):
                dr, sm, sr = d[i], s_m[i], s_r[i]
                true_mid   = q_m[i].item()

                scores = vectorized_scores(model, dr, sm, sr)

                # apply ranking method
                if method=="raw":
                    sco = scores
                elif method=="residual":
                    sco = scores - pop_baseline
                elif method=="popularity":
                    sco = counts.to(device)
                elif method=="zscore":
                    sco = (scores - raw_means.to(device)) / (stddevs.to(device)+1e-8)
                else:
                    raise ValueError(f"Unknown method {method}")

                # optional hard‐filter
                if filter_count is not None:
                    mask = counts >= filter_count
                    sco  = torch.where(mask.to(device),
                                       sco,
                                       torch.full_like(sco, -1e6))

                top5 = torch.topk(sco, K).indices.tolist()
                if true_mid in top5:
                    hits += 1
                total += 1

    return hits/total if total>0 else 0.0


# ─── 3) main: loop over each support‐set size and each method ───────────
def main():
    cfg = load_config("config.json")

    # 3.1) full ratings + demographics
    ratings_df, demog_tensor, n_users, n_items = gen_demographic_table(
        config_path="config.json",
        verbose=cfg.get("verbose",False),
    )

    # precompute per-item stats
    counts_series = ratings_df.groupby("mid").size().reindex(range(n_items), fill_value=0)
    counts     = torch.tensor(counts_series.values, dtype=torch.float)
    gm = ratings_df.groupby("mid")["Rating"].mean().reset_index()
    idxs = torch.tensor(gm.mid.values,    dtype=torch.long)   # item indices
    vals = torch.tensor(gm.Rating.values, dtype=torch.float)  # mean ratings

    raw_means = torch.zeros(n_items, dtype=torch.float)
    raw_means[idxs] = vals

    std_series = (
        ratings_df
        .groupby("mid")["Rating"]
        .std()
        .reindex(range(n_items), fill_value=0.0)
    )
    stddevs = torch.tensor(std_series.values, dtype=torch.float)
    overall    = float(ratings_df.Rating.mean())

    # 3.2) NeuMF tables
    item_gmf, item_mlp = load_save_neumf_table("config.json")

    # 3.3) rebuild & freeze HyperNeuMF
    ckpt  = Path(cfg["app"]["ckpt_to_use"]).expanduser().resolve()
    model = DCHyperNeuMF.load_from_checkpoint(str(ckpt))
    model.item_gmf.weight.data.copy_(item_gmf)
    model.item_mlp.weight.data.copy_(item_mlp)
    for p in (list(model.item_gmf.parameters()) + list(model.item_mlp.parameters())):
        p.requires_grad_(False)
    model.eval()

    # 3.4) evaluate
    print("\nMethod      |   RMSE   |  P@5  ")
    print("-----------------------------------")
    for k in cfg["model_hyperparams"]["k"]:
        dm,_,_,_ = gen_datamodule(
            k_list   = [k],
            max_k    = k,
            test_size=0.2,
            random_state=42,
            verbose=False,
        )
        rmse = compute_rmse(model, dm, device="cpu")

        for method in ["raw","residual","popularity","zscore"]:
            prec = compute_precision_at_5(
                model, dm,
                counts=counts,
                raw_means=raw_means,
                stddevs=stddevs,
                overall_mean=overall,
                K=5,
                method=method,
                shrinkage=True,
                filter_count=10,
                device="cpu",
            )
            print(f"k={k:2d}, {method:10s} → {rmse:6.3f} | {prec:5.3f}")
        print()

if __name__=="__main__":
    main()
