import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


neutral_rating = 3.5  

class SupportSetEncoder(nn.Module):
    """
    Compress k (movie, rating) pairs into a single vector r_u ∈ ℝ^d,
    with sharpened weighting of extreme ratings.
    """
    def __init__(self, item_emb: nn.Embedding, exponent: float = 1.5):
        super().__init__()
        self.item_emb = item_emb          
        self.exponent = exponent

    def forward(self, movie_ids, ratings):
        batch, k = movie_ids.shape
        if k == 0:
            d = self.item_emb.embedding_dim
            return torch.zeros(batch, d, device=movie_ids.device)

        e = self.item_emb(movie_ids)               

        w0 = (ratings - neutral_rating).unsqueeze(-1) 

        w = torch.sign(w0) * torch.abs(w0) ** self.exponent 

        norm = w.abs().sum(dim=1, keepdim=True).clamp(min=1.0) 
        w = w / norm

        r_u = (w * e).sum(dim=1)  
        return r_u