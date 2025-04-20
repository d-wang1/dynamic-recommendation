import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


neutral_rating = 3.5  # center point

class SupportSetEncoder(nn.Module):
    """
    Compress k (movie, rating) pairs into a single vector r_u ∈ ℝ^d,
    with sharpened weighting of extreme ratings.
    """
    def __init__(self, item_emb: nn.Embedding, exponent: float = 1.5):
        super().__init__()
        self.item_emb = item_emb          # frozen movie table
        self.exponent = exponent

    def forward(self, movie_ids, ratings):
        # movie_ids: (batch, k), ratings: (batch, k)
        batch, k = movie_ids.shape
        if k == 0:
            # zero vibe vector for cold‑start
            d = self.item_emb.embedding_dim
            return torch.zeros(batch, d, device=movie_ids.device)

        # Embeddings for each support movie
        e = self.item_emb(movie_ids)               # (batch, k, d)

        # 1) center around neutral
        w0 = (ratings - neutral_rating).unsqueeze(-1)  # (batch, k, 1)

        # 2) amplify extremes
        w = torch.sign(w0) * torch.abs(w0) ** self.exponent  # (batch, k, 1)

        # 3) normalize so sum of abs weights = 1
        norm = w.abs().sum(dim=1, keepdim=True).clamp(min=1.0)  # (batch, 1, 1)
        w = w / norm

        # 4) weighted sum of embeddings
        r_u = (w * e).sum(dim=1)  # (batch, d)
        return r_u