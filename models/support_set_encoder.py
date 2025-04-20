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
        # movie_ids: (batch, k), ratings: (batch, k)
        batch, k = movie_ids.shape
        if k == 0:
            # return a zero "vibe" vector for every batch row
            d = self.item_emb.embedding_dim
            return torch.zeros(batch, d, device=movie_ids.device)
        # else do the usual weighted average
        e = self.item_emb(movie_ids)                   # (batch, k, d)
        w = (ratings - 3.5).unsqueeze(-1)               # center
        return (w * e).mean(dim=1) 