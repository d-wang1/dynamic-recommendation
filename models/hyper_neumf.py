import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models.support_set_encoder import SupportSetEncoder
from torchmetrics import MeanSquaredError


class DCHyperNeuMF(pl.LightningModule):
    """
    Demographic‑Conditioned Hypernetwork that prints user embeddings
    and feeds them into a slim NeuMF rating head.
    """
    def __init__(
        self,
        n_movies: int,
        d_emb: int = 8,
        demog_dim: int = 32,
        mlp_layers=(64, 32, 16),
        lr: float = 1e-3,
        encoder_exponent: float = 1.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Movie embeddings. For each movie id, store a linear fingerprint (len d_emb) in GMF and a nonlinear one in MLP.
        self.item_gmf = nn.Embedding(n_movies, d_emb)
        self.item_mlp = nn.Embedding(n_movies, d_emb)

        # Later load pre‑trained weights and freeze them: (UPDATE: Doing this in train.py)
        # for p in self.item_gmf.parameters(): p.requires_grad_(False)

        # Support Set Encoder (SSE) for k (movie, rating) pairs as a "vibe" vector
        self.supenc = SupportSetEncoder(self.item_mlp, exponent=encoder_exponent)

        # Forge a lens from "vibe" and demographic vector
        in_dim = demog_dim + d_emb

        hid = 2 * d_emb
        # Simple small hypernetwork that takes in a demographic vector and a "vibe" vector (from SSE) and produces two vectors of size d_emb each (separate later).
        # The first one is used in the GMF part of the model, and the second one is used in the MLP part of the model.
        self.hyper = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, 2 * d_emb),
        )

        # Nonlinear MLP head
        mlp_units = []
        # Takes in both user and movie embeddings
        inp = 2 * d_emb
        # Create network based on layer sizes
        for h in mlp_layers:
            mlp_units += [nn.Linear(inp, h), nn.ReLU()]
            inp = h
        # Dense layer + RELU each time
        self.mlp_head = nn.Sequential(*mlp_units)
        # Join GMF and MLP for one singular prediction score

        # Scalar GMF option
        # self.out = nn.Linear(1 + mlp_layers[-1], 1)

        # Vector GMF option
        self.gate = nn.Sequential(
            nn.Linear(d_emb + mlp_layers[-1], 1),
            nn.Sigmoid()
        )
        # 2) GMF‐only output (no bias, since popularity biases are baked into embeddings)
        self.out_gmf = nn.Linear(d_emb, 1, bias=False)
        # 3) MLP‐only output (with bias)
        self.out_mlp = nn.Linear(mlp_layers[-1], 1, bias=True)

        self.lr = lr

    # "First impression" vector 
    def make_user_embs(self, demog_vec, movie_ids, ratings):
        """
        demog_vec : (batch, demog_dim)
        movie_ids, ratings : (batch, k)
        returns u_gmf, u_mlp  each (batch, d_emb)
        """
        # SSE to determine user's preferences based on k (movie, rating) pairs
        r_vec = self.supenc(movie_ids, ratings)            # (batch, d_emb)
        # Combine demographic and preference information
        h_inp = torch.cat([demog_vec, r_vec], dim=1)
        # Create lens to use in GMF and MLP parts of the model
        u = self.hyper(h_inp)                              # (batch, 2d)
        u_gmf, u_mlp = u.chunk(2, dim=1)
        return u_gmf, u_mlp

    # ----- forward --------------------------------------------------------
    def forward(self, demog_vec, supp_movies, supp_ratings, query_movies):
        """
        Produce *predicted* ratings for query_movies.
        demog_vec    : (batch, demog_dim)
        supp_movies  : (batch, k)
        supp_ratings : (batch, k)
        query_movies : (batch,)      1 movie per row for simplicity
        """
        # 1) make user embs
        u_gmf, u_mlp = self.make_user_embs(demog_vec, supp_movies, supp_ratings)

        # 2) pull item embs
        v_gmf = self.item_gmf(query_movies)
        v_mlp = self.item_mlp(query_movies)

        # 3) GMF branch vector
        gmf_vec = u_gmf * v_gmf                        # (batch, d_emb)
        # 4) MLP branch vector
        mlp_feats = self.mlp_head(torch.cat([u_mlp, v_mlp], dim=1))  # (batch, mlp_last)

        # 5) compute gate α ∈ [0,1]
        gate_in = torch.cat([gmf_vec, mlp_feats], dim=1)  # (batch, d_emb+mlp_last)
        alpha = self.gate(gate_in)                            # (batch, 1)
        self.alpha = alpha.squeeze(1)                          # (batch,)
        # 6) each branch’s scalar score
        s_gmf = self.out_gmf(gmf_vec)      # (batch, 1)
        s_mlp = self.out_mlp(mlp_feats)    # (batch, 1)

        # 7) combine with learned gate
        pred = alpha * s_gmf + (1 - alpha) * s_mlp  # (batch, 1)
        return pred.squeeze(1)             # (batch,)

    def training_step(self, batch, _):
        d, supp_m, supp_r, q_m, q_r = batch   # unpack
        y_hat = self(d, supp_m, supp_r, q_m)
        loss = F.mse_loss(y_hat, q_r)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, _):
        d, supp_m, supp_r, q_m, q_r = batch
        y_hat = self(d, supp_m, supp_r, q_m)
        rmse = torch.sqrt(F.mse_loss(y_hat, q_r))
        self.log("val_rmse", rmse, prog_bar=True, sync_dist=True)
        self.log("gate_mean", self.alpha.mean(), prog_bar=True)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)