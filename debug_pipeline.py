# debug_pipeline.py
import torch
from data.helper import gen_datamodule, load_save_neumf_table, load_config
from models.hyper_neumf import DCHyperNeuMF
from data.objs import AGE2IDX
import torch.nn.functional as F
from data.helper import gen_demographic_table
from pathlib import Path

def load_everything():
    cfg = load_config("config.json")
    # 1) build datamodule with k=3
    dm, n_users, n_movies, demog = gen_datamodule(k=3,
                                                  test_size=0.2,
                                                  random_state=42,
                                                  verbose=False)
    # 2) load pretrained tables
    item_gmf, item_mlp = load_save_neumf_table("config.json")
    # 3) rebuild model
    ckpt = cfg["app"]["ckpt_to_use"]
    model = DCHyperNeuMF.load_from_checkpoint(ckpt, strict=False)
    model.item_gmf.weight.data.copy_(item_gmf)
    model.item_mlp.weight.data.copy_(item_mlp)
    for p in model.item_gmf.parameters():
        p.requires_grad_(False)
    for p in model.item_mlp.parameters():
        p.requires_grad_(False)
    model.eval()
    return dm, demog, model

def test_dataloader(dm):
    print("==> Testing dataloader shapes & one batch")
    loader = dm.val_dataloader()
    batch = next(iter(loader))
    d, s_m, s_r, q_m, q_r = batch
    print("demog shape      ", d.shape)
    print("demog values     ", d[2,:])
    print("support movies   ", s_m.shape)
    print("support movies  ", s_m[0,:])
    print("support ratings  ", s_r.shape)
    print("support ratings ", s_r[0,:])
    print("query movies     ", q_m.shape)
    # print("query movies    ", q_m)
    print("query ratings    ", q_r.shape)
    # print("query ratings   ", q_r)
    # are any supports length zero?
    zeros = (s_m.sum(dim=1)==0).sum().item()
    print("zero‐support rows:", zeros, "/", d.size(0))

def test_support_encoder(model, demog):
    print("\n==> Testing SupportSetEncoder")
    enc = model.supenc
    # try k=0
    empty_m = torch.empty((2,0), dtype=torch.long)
    empty_r = torch.empty((2,0), dtype=torch.float)
    z = enc(empty_m, empty_r)
    print(" k=0 →", z.shape, " all zeros? ", bool(z.abs().sum().item()==0))
    # try random k=3
    mids = torch.randint(0, model.item_mlp.num_embeddings, (2,3))
    rates = torch.rand((2,3))*4+1
    y = enc(mids, rates)
    print(" k=3 →", y.shape, " sample:", y[0,:5])

def test_forward_batch(dm, demog, model):
    print("\n==> Testing full forward on a handful of users")
    loader = dm.val_dataloader()
    batch = next(iter(loader))
    d, s_m, s_r, q_m, q_r = batch
    # pick 3 random rows
    for i in [0,1,2]:
        di = d[i].unsqueeze(0)
        sm = s_m[i].unsqueeze(0)
        sr = s_r[i].unsqueeze(0)
        qm = q_m[i].unsqueeze(0)
        # forward
        with torch.no_grad():
            out = model(di, sm, sr, qm)
        print(f" user {i} → forward output type: {type(out)} ; values:", out[:5])

def scan_gate_weights(dm, demog, model):
    print("\n==> Scanning gate weights over entire val set")
    loader = dm.val_dataloader()
    all_ws = []
    for batch in loader:
        d, s_m, s_r, q_m, q_r = batch
        with torch.no_grad():
            _, wts = model(d, s_m, s_r, q_m)
        all_ws.append(wts.cpu())
    all_ws = torch.cat(all_ws, dim=0)
    print(" gate weights shape:", all_ws.shape)
    print("  avg per branch:", all_ws.mean(dim=0))
    print("   min / max per branch:", all_ws.min(dim=0)[0], all_ws.max(dim=0)[0])

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

def compare_vector_vs_forward(model, demog_row, supp_m, supp_r, query_mid):
    """
    Compare model.forward on one movie vs. our vectorized_scores over all movies.
    Prints and returns:
      - direct_pred  : float
      - vector_pred  : float
      - direct_gates : [g_gmf, g_mlp, g_demog]
      - vector_gates : [g_gmf, g_mlp, g_demog]
    """
    model.eval()
    device = demog_row.device

    # 1) Direct single‐movie call
    #    demog_row: (demog_dim,) → (1, demog_dim)
    #    supp_m, supp_r: each (k,) → (1, k)
    #    query_mid: single int → (1,)
    direct_out = model(
        demog_row.unsqueeze(0),
        supp_m.unsqueeze(0),
        supp_r.unsqueeze(0),
        torch.tensor([query_mid], device=device)
    )
    if isinstance(direct_out, tuple):
        direct_pred_tensor, direct_gates_tensor = direct_out
    else:
        direct_pred_tensor, direct_gates_tensor = direct_out, None

    direct_pred = direct_pred_tensor.item()
    direct_gates = direct_gates_tensor.squeeze(0).tolist() if direct_gates_tensor is not None else None

    # 2) Vectorized over all N movies
    all_preds, all_gates = vectorized_scores(model, demog_row, supp_m, supp_r)

    vector_pred = all_preds[query_mid].item()
    vector_gates = all_gates[query_mid].tolist() if all_gates is not None else None

    # 3) Report
    print(f"\n=== Comparing for query_mid={query_mid} ===")
    print(f"Direct   pred: {direct_pred:.6f}")
    print(f"Vector   pred: {vector_pred:.6f}")
    print(f"Direct   gates: {direct_gates}")
    print(f"Vector   gates: {vector_gates}\n")

    return direct_pred, vector_pred, direct_gates, vector_gates


def main():
    cfg = load_config("config.json")
    item_gmf, item_mlp = load_save_neumf_table(
        config_path="config.json",
        verbose=cfg.get("verbose", False),
    )

    # --- rebuild & freeze your hyper-model ---
    ckpt_path = Path(cfg["app"]["ckpt_to_use"]).expanduser().resolve()
    model: DCHyperNeuMF = DCHyperNeuMF.load_from_checkpoint(str(ckpt_path))

    # copy in the frozen NeuMF tables
    model.item_gmf.weight.data.copy_(item_gmf)
    model.item_mlp.weight.data.copy_(item_mlp)
    for p in model.item_gmf.parameters():
        p.requires_grad_(False)
    for p in model.item_mlp.parameters():
        p.requires_grad_(False)
    model.eval()

    # --- load ratings & demographics ---
    ratings_df, demog_tensor, n_users, n_items = gen_demographic_table(
        config_path="config.json",
        verbose=cfg.get("verbose", False),
    )
    print(f"Loaded {n_users} users, {n_items} movies, {len(ratings_df)} ratings.")
    

    
    # --- pick a toy user & support set ---
    uid = 0
    user_hist = ratings_df[ratings_df.uid==uid] \
                .sort_values("Timestamp", ascending=False)
    supp       = user_hist.head(5)
    print(f"Support set for user {uid}:\n", supp)
    supp_m = torch.tensor(supp.mid.values, dtype=torch.long)
    supp_r = torch.tensor(supp.Rating.values, dtype=torch.float)
    demog_row = demog_tensor[uid]

    # --- pick a query movie (e.g. the (k+1)-th that user saw) ---
    # query_mid = int(user_hist.iloc[k].mid)
    query_mid = 318
    pred_direct, gates = model(
    demog_row.unsqueeze(0),
    supp_m.unsqueeze(0),
    supp_r.unsqueeze(0),
    torch.tensor([query_mid])
    )
    print("DEBUG direct → ", pred_direct.item(), "gates → ", gates)
    # --- run the compare ---
    compare_vector_vs_forward(model, demog_row, supp_m, supp_r, query_mid)



if __name__ == "__main__":
    main()
