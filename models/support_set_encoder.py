import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


neutral_rating = 3.5  # neutral rating for movies (1-5 scale). REMINDER: Use 3.5 or 3?
class SupportSetEncoder(nn.Module):
    """
    Compress k (movie, rating) pairs into a single vector r_u ∈ ℝ^d.
    """
    def __init__(self, item_emb: nn.Embedding):
        super().__init__()
        self.item_emb = item_emb          # frozen movie table

    def forward(self, movie_ids, ratings):
        """
        movie_ids : (batch, k)  int64
        ratings   : (batch, k)  float32   values 1–5
        """
        # (batch, k, d)
        e = self.item_emb(movie_ids)

        w = (ratings - neutral_rating).unsqueeze(-1)

        # weighted average → (batch, d)
        r = (w * e).mean(dim=1)
        return r