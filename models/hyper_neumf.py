import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models.support_set_encoder import SupportSetEncoder


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
    ):
        super().__init__()
        self.save_hyperparameters()

        # Movie embeddings. For each movie id, store a linear fingerprint (len d_emb) in GMF and a nonlinear one in MLP.
        self.item_gmf = nn.Embedding(n_movies, d_emb)
        self.item_mlp = nn.Embedding(n_movies, d_emb)

        # TODO: Later load pre‑trained weights and freeze them:
        # for p in self.item_gmf.parameters(): p.requires_grad_(False)

        # Support Set Encoder (SSE) for k (movie, rating) pairs as a "vibe" vector
        self.supenc = SupportSetEncoder(self.item_mlp)

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
        self.out = nn.Linear(d_emb + mlp_layers[-1], 1)

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
        u_gmf, u_mlp = self.make_user_embs(
            demog_vec, supp_movies, supp_ratings
        )
        v_gmf = self.item_gmf(query_movies)
        v_mlp = self.item_mlp(query_movies)

        gmf = (u_gmf * v_gmf).sum(dim=1, keepdim=True)     # (batch,1)
        mlp = self.mlp_head(torch.cat([u_mlp, v_mlp], 1))  # (batch, h)
        pred = self.out(torch.cat([gmf, mlp], 1)).squeeze(1)  # (batch,)
        return pred                                          # raw score

    def training_step(self, batch, _):
        d, supp_m, supp_r, q_m, q_r = batch   # unpack
        y_hat = self(d, supp_m, supp_r, q_m)
        loss = F.mse_loss(y_hat, q_r)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)