import torch
import pandas as pd
from data.helper import gen_demographic_table, load_ratings, load_save_neumf_table
from models.hyper_neumf import DCHyperNeuMF
from data.objs import AGE2IDX, AGE_BUCKET_CODES

# 1) Paths & hyperparams (adjust to your config)
CKPT = "checkpoints/epochepoch=80-rmseval_rmse=0.956.ckpt"
NEUMF_TABLE = "neumf_movie_tables.pt"
MOVIES_FILE = "./ml-1m/movies.dat"
RATINGS_FILE = "./ml-1m/ratings.dat"
USERS_FILE   = "./ml-1m/users.dat"

# 2) Load model + frozen tables
model: DCHyperNeuMF = DCHyperNeuMF.load_from_checkpoint(CKPT)
tables = torch.load(NEUMF_TABLE, map_location="cpu")
model.item_gmf.weight.data.copy_(tables["item_gmf"])
model.item_mlp.weight.data.copy_(tables["item_mlp"])
for p in model.item_gmf.parameters():
    p.requires_grad_(False)
for p in model.item_mlp.parameters():
    p.requires_grad_(False)
model.eval()

# 3) Load metadata
ratings_df, demog_tensor, n_users, n_items = gen_demographic_table()
raw_ratings = pd.read_csv(RATINGS_FILE, sep="::", engine="python",
                          names=["UserID","MovieID","Rating","Timestamp"], encoding="ISO-8859-1")
# map raw→internal IDs
mid_map = {m:i for i,m in enumerate(raw_ratings.MovieID.unique())}
movies = pd.read_csv(MOVIES_FILE, sep="::", engine="python",
                     names=["MovieID","Title","Genres"], encoding="ISO-8859-1")
titles = [None]*n_items
for _,r in movies.iterrows():
    idx = mid_map.get(r.MovieID)
    if idx is not None:
        titles[idx] = r.Title

# 4) helper: vectorized_scores + gate extraction
def vector_scores_and_gates(model, d_row):
    # zero‑shot support
    supp_m = torch.empty((0,), dtype=torch.long)
    supp_r = torch.empty((0,), dtype=torch.float)
    with torch.no_grad():
        # 1) user vectors
        u_gmf, u_mlp = model.make_user_embs(
            d_row.unsqueeze(0),         # (1, demog_dim)
            supp_m.unsqueeze(0),        # (1,0)
            supp_r.unsqueeze(0),        # (1,0)
        )  # both (1, d_emb)

        # 2) pull item tables
        V_gmf   = model.item_gmf.weight    # (N, d_emb)
        V_mlp   = model.item_mlp.weight    # (N, d_emb)
        V_demog = model.item_demog.weight  # (N, demog_feat_dim)

        # 3) GMF branch
        gmf_vec = (u_gmf * V_gmf)          # broadcasts to (N, d_emb)
        s_gmf   = model.out_gmf(gmf_vec).squeeze(1)  # (N,)

        # 4) MLP branch
        um       = u_mlp.repeat(len(V_mlp), 1)               # (N, d_emb)
        mlp_in   = torch.cat([um, V_mlp], dim=1)             # (N, 2*d_emb)
        mlp_feats= model.mlp_head(mlp_in)                    # (N, mlp_last)
        s_mlp    = model.out_mlp(mlp_feats).squeeze(1)       # (N,)

        # 5) Demographic branch
        # rebuild demog embedding for this user, then repeat
        g = model.gender_emb(d_row[[0]])
        raw_age = d_row[1].item()
        age_idx = AGE2IDX.get(raw_age, 0)
        a = model.age_emb(torch.tensor([age_idx], device=d_row.device))
        o = model.occ_emb(d_row[2].unsqueeze(0))
        z = model.zip_emb(d_row[3].unsqueeze(0))
        dem_feat = torch.cat([g,a,o,z], dim=1)           # (1, demog_feat_dim)
        dem_rep  = dem_feat.repeat(len(V_demog), 1)      # (N, demog_feat_dim)
        dem_in   = dem_rep * V_demog                     # (N, demog_feat_dim)
        s_demog  = model.out_demog(dem_in).squeeze(1)    # (N,)

        # 6) gate & fuse
        gate_in = torch.cat([gmf_vec, mlp_feats, dem_in], dim=1)  # (N, sum_dim)
        wts     = model.gate3(gate_in)                            # (N,3)
        fused   = wts[:,0]*s_gmf + wts[:,1]*s_mlp + wts[:,2]*s_demog  # (N,)

        return fused, wts

# 5) build two demog rows
# find raw code → you can adjust these keys if different
student = [0, 1, 10, 63]  # M, bucket=1, occ=10, zip=63
retired = [1, 56, 13, 63] # F, bucket=56, occ=13, zip=63
# turn into tensors
d_student = torch.tensor(student, dtype=torch.long)
d_retired = torch.tensor(retired, dtype=torch.long)

# 6) score both
pred_s, gates_s = vector_scores_and_gates(model, d_student)
pred_r, gates_r = vector_scores_and_gates(model, d_retired)

# 7) top‑20 lists
top_s = pred_s.topk(20).indices.tolist()
top_r = pred_r.topk(20).indices.tolist()

# 8) Jaccard similarity
set_s, set_r = set(top_s), set(top_r)
jaccard = len(set_s & set_r) / len(set_s | set_r)

# 9) print results
print("\n=== Student Top‑20 ===")
for i in top_s:
    print(titles[i], f"({pred_s[i]:.2f})")
print("\n=== Retired Top‑20 ===")
for i in top_r:
    print(titles[i], f"({pred_r[i]:.2f})")
print(f"\nJaccard similarity between the two lists: {jaccard:.2f}")

print("\n--- Avg. gate weights (student) ---")
avg_gates = gates_s.mean(dim=0)       # tensor([gmf_mean, mlp_mean, demog_mean])
g0, g1, g2 = avg_gates.tolist()
print(f"Avg Gate Weights — GMF: {g0:.3f}, MLP: {g1:.3f}, DEMOG: {g2:.3f}")
print(f"GMF: {g0:.3f}, MLP: {g1:.3f}, DEMOG: {g2:.3f}")
print("\n--- Avg. gate weights (retired) ---")
avg_gates = gates_r.mean(dim=0)       # tensor([gmf_mean, mlp_mean, demog_mean])
g0, g1, g2 = avg_gates.tolist()
print(f"Avg Gate Weights — GMF: {g0:.3f}, MLP: {g1:.3f}, DEMOG: {g2:.3f}")
print(f"GMF: {g0:.3f}, MLP: {g1:.3f}, DEMOG: {g2:.3f}")