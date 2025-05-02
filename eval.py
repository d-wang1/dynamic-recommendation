import argparse
import math
import torch
from pathlib import Path

from data.helper import (
    load_config,
    gen_datamodule,
    load_save_neumf_table,
)
from models.hyper_neumf import DCHyperNeuMF

from data.objs import AGE2IDX
import torch.nn.functional as F

def vectorized_scores(model, demog_row, supp_m, supp_r):
    """
    Score every movie for one user in one go using the GMF, MLP, and Demog branches + gating.
    demog_row: LongTensor of shape (4,)  [gender, age_code, occ, zip]
    supp_m, supp_r: LongTensors of shape (k,)
    Returns: FloatTensor of shape (n_items,) of raw predictions.
    """
    with torch.no_grad():
        device = demog_row.device
        # 1) user embeddings
        u_gmf, u_mlp = model.make_user_embs(
            demog_row.unsqueeze(0),   # → (1, demog_dim)
            supp_m.unsqueeze(0),      # → (1, k)
            supp_r.unsqueeze(0),      # → (1, k)
        )                             # each is (1, d_emb)

        # 2) frozen item tables
        V_gmf   = model.item_gmf.weight      # (n_items, d_emb)
        V_mlp   = model.item_mlp.weight      # (n_items, d_emb)
        V_demog = model.item_demog.weight    # (n_items, demog_feat_dim)

        # 3) GMF branch
        gmf_vec = u_gmf * V_gmf               # (n_items, d_emb)
        s_gmf   = model.out_gmf(gmf_vec).squeeze(1)    # (n_items,)

        # 4) MLP branch
        um       = u_mlp.repeat(len(V_mlp), 1)
        mlp_feat = model.mlp_head(torch.cat([um, V_mlp], dim=1))  # (n_items, mlp_last)
        s_mlp    = model.out_mlp(mlp_feat).squeeze(1)             # (n_items,)

        # 5) Demog branch — inline exactly your forward’s code:
        #   a) build one‐row demog_feats
        g = model.gender_emb(demog_row[[0]])  # (1, d_emb)
        raw_age = demog_row[1].item()
        age_idx = AGE2IDX.get(raw_age, 0)
        a = model.age_emb(torch.tensor([age_idx], device=device))
        o = model.occ_emb(demog_row[[2]])
        z = model.zip_emb(demog_row[[3]])
        demog_feats_1 = torch.cat([g, a, o, z], dim=1)  # (1, demog_feat_dim)

        #   b) replicate to all items, multiply by item_demog table
        dem_feats = demog_feats_1.repeat(len(V_demog), 1)  # (n_items, demog_feat_dim)
        dem_in    = dem_feats * V_demog                   # (n_items, demog_feat_dim)
        s_demog   = model.out_demog(dem_in).squeeze(1)    # (n_items,)

        # 6) normalize each branch before gating
        n_g = model.norm_gmf(gmf_vec)
        n_m = model.norm_mlp(mlp_feat)
        n_d = model.norm_demog(dem_in)

        # 7) k‑feature for gate
        k = supp_m.numel()
        max_k = getattr(model.hparams, "max_k", k)
        k_feat = torch.full((len(V_gmf),1),
                              fill_value=k/float(max_k),
                              device=device)

        # 8) build gate inputs & compute weights
        gate_in = torch.cat([n_g, n_m, n_d, k_feat], dim=1)   # (n_items, total_dim+1)
        logits  = model.gate_linear(gate_in)                  # (n_items, 3)
        wts     = F.softmax(logits / model.temperature, dim=1) # (n_items, 3)

        # 9) fuse branch scores
        fused = wts[:,0]*s_gmf + wts[:,1]*s_mlp + wts[:,2]*s_demog  # (n_items,)

        return fused


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",    type=Path, help="path to .ckpt", required=True)
    p.add_argument("--batch",   type=int,  default=256)
    p.add_argument("--k",       type=int,  default=3,
                   help="support‐set size used at training")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_config("config.json")

    # 1) load pretrained NeuMF tables
    item_gmf, item_mlp = load_save_neumf_table(
        config_path="config.json",
        verbose=cfg.get("verbose", False),
    )

    # 2) rebuild & freeze your HyperNeuMF
    model: DCHyperNeuMF = DCHyperNeuMF.load_from_checkpoint(str(args.ckpt))
    model.item_gmf.weight.data.copy_(item_gmf)
    model.item_mlp.weight.data.copy_(item_mlp)
    for p in model.item_gmf.parameters():
        p.requires_grad_(False)
    for p in model.item_mlp.parameters():
        p.requires_grad_(False)
    model.eval()

    # 3) instantiate your k‑shot datamodule (using the same k & split as training)
    dm, n_users, n_movies, _ = gen_datamodule(
        k=args.k,
        test_size=0.2,
        random_state=42,
        verbose=cfg.get("verbose", False),
    )
    test_loader = dm.val_dataloader()  # or dm.test_dataloader() if you have one

    # 4) metrics accumulators
    hits, ndcgs, mrrs = [], [], []

    # 5) loop
    for demog, supp_m, supp_r, q_m, q_r in test_loader:
        B = demog.size(0)
        for i in range(B):
            scores = vectorized_scores(model,
                                       demog[i],
                                       supp_m[i],
                                       supp_r[i])              # (n_movies,)

            # 5a) mask out the support set
            if args.k > 0:
                scores[supp_m[i]] = -1e9

            # 5b) get top‑5
            top5 = torch.topk(scores, k=5).indices.tolist()
            pos  = q_m[i].item()

            # Precision@5
            hits.append(1.0 if pos in top5 else 0.0)

            # find full ranking position
            sorted_idx = torch.argsort(scores, descending=True)
            rank_pos   = (sorted_idx == pos).nonzero(as_tuple=False)[0].item()

            # NDCG@5
            ndcgs.append((1.0 / math.log2(rank_pos+2)) if rank_pos < 5 else 0.0)

            # MRR
            mrrs.append(1.0 / (rank_pos+1))

    # 6) final
    P5   = sum(hits)  / len(hits)
    NDCG5= sum(ndcgs) / len(ndcgs)
    MRR  = sum(mrrs)  / len(mrrs)
    print(f"\nPrecision@5: {P5:.4f}")
    print(f" NDCG@5:     {NDCG5:.4f}")
    print(f" MRR:        {MRR:.4f}")

if __name__ == "__main__":
    main()