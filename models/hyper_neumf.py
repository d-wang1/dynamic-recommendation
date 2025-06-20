import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models.support_set_encoder import SupportSetEncoder
from torchmetrics import MeanSquaredError
from data.objs import AGE2IDX


class DCHyperNeuMF(pl.LightningModule):
    """
    Demographic‑Conditioned Hypernetwork that prints user embeddings
    and feeds them into a slim NeuMF rating head.
    """
    def __init__(
        self,
        n_movies: int,
        d_emb: int = 8,
        demog_dim: int = 4,       
        mlp_layers=(64,32,16),
        lr: float = 1e-3,
        max_k: int = 3,           
    ):
        super().__init__()
        self.save_hyperparameters("n_movies","d_emb","demog_dim","mlp_layers","lr","max_k")

        self.item_gmf = nn.Embedding(n_movies, d_emb)
        self.item_mlp = nn.Embedding(n_movies, d_emb)
        self.supenc    = SupportSetEncoder(self.item_mlp)
        demog_emb_dim = d_emb
        self.gender_emb = nn.Embedding(2, demog_emb_dim)
        self.age_emb    = nn.Embedding(7, demog_emb_dim)
        self.occ_emb    = nn.Embedding(21, demog_emb_dim)
        self.zip_emb    = nn.Embedding(100, demog_emb_dim)

        demog_feat_dim = demog_emb_dim * 4
        in_dim = demog_feat_dim + d_emb
        hid    = 2 * d_emb
        self.hyper = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, 2*d_emb),
        )

        mlp_units = []
        inp = 2 * d_emb
        for h in mlp_layers:
            mlp_units += [nn.Linear(inp, h), nn.ReLU()]
            inp = h
        self.mlp_head = nn.Sequential(*mlp_units)

        self.out_gmf = nn.Linear(d_emb, 1, bias=False)
        self.out_mlp = nn.Linear(mlp_layers[-1], 1, bias=True)

        self.item_demog = nn.Embedding(n_movies, demog_feat_dim)
        self.out_demog = nn.Linear(demog_feat_dim, 1, bias=True)

        self.demog_bias = nn.Linear(demog_feat_dim, 1, bias=True)


        total_dim = d_emb + mlp_layers[-1] + demog_feat_dim
        self.gate_linear = nn.Linear(total_dim + 1, 3)
        nn.init.constant_(self.gate_linear.bias, 0.0)
        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.norm_gmf   = nn.LayerNorm(d_emb)
        self.norm_mlp   = nn.LayerNorm(mlp_layers[-1])
        self.norm_demog = nn.LayerNorm(demog_feat_dim)

        self.max_k = max_k
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

        g = self.gender_emb(demog_vec[:, 0])
        raw_ages = demog_vec[:, 1].tolist()   
        # 2) map each code to its 0–6 index
        age_idxs = [AGE2IDX.get(int(a), 0) for a in raw_ages]
        # 3) turn that back into a tensor on the correct device
        age_tensor = torch.tensor(age_idxs, device=demog_vec.device, dtype=torch.long)
        # 4) lookup your embedding
        a = self.age_emb(age_tensor)
        o = self.occ_emb   (demog_vec[:, 2])
        z = self.zip_emb   (demog_vec[:, 3])
        demog_feat = torch.cat([g, a, o, z], dim=1)  # (batch, 4*demog_emb_dim)
        # Combine demographic embedding and “vibe”
        h_inp = torch.cat([demog_feat, r_vec], dim=1)
        # Create lens to use in GMF and MLP parts of the model
        u = self.hyper(h_inp)                              # (batch, 2d)
        u_gmf, u_mlp = u.chunk(2, dim=1)
        return u_gmf, u_mlp

    def forward(self, demog_vec, supp_m, supp_r, query_m):
        B, k = supp_m.size()
        k_feat = (k / float(self.hparams.max_k))
        k_feat = k_feat * torch.ones(B,1, device=demog_vec.device)
        u_gmf, u_mlp = self.make_user_embs(demog_vec, supp_m, supp_r)

        v_gmf = self.item_gmf(query_m)
        v_mlp = self.item_mlp(query_m)

        gmf_vec = u_gmf * v_gmf              
        s_gmf   = self.out_gmf(gmf_vec)      

        mlp_feat = self.mlp_head(torch.cat([u_mlp, v_mlp], dim=1))  
        s_mlp    = self.out_mlp(mlp_feat)                          

        g = self.gender_emb(demog_vec[:,0])
        raw_ages = demog_vec[:, 1].tolist()
        age_idxs = [AGE2IDX.get(int(a), 0) for a in raw_ages]
        age_tensor = torch.tensor(age_idxs, device=demog_vec.device, dtype=torch.long)
        a = self.age_emb(age_tensor)
        o = self.occ_emb(demog_vec[:,2])
        z = self.zip_emb(demog_vec[:,3])
        dem_feats = torch.cat([g,a,o,z], dim=1)       
        v_demog   = self.item_demog(query_m)          
        dem_in    = dem_feats * v_demog               
        s_demog   = self.out_demog(dem_in)            

        gmf_norm   = self.norm_gmf(gmf_vec)    
        mlp_norm   = self.norm_mlp(mlp_feat)   
        dem_norm   = self.norm_demog(dem_in)   

        gate_in    = torch.cat([gmf_norm, mlp_norm, dem_norm, k_feat], dim=1)
        logits     = self.gate_linear(gate_in)                         
        wts        = F.softmax(logits / self.temperature, dim=1)       

        scores = torch.cat([s_gmf, s_mlp, s_demog], dim=1)   
        base   = (wts * scores).sum(dim=1)                   

        b_demog = self.demog_bias(dem_feats).squeeze(1)      


        pred = base + b_demog                                
        return pred, wts

    def training_step(self, batch, batch_idx):
        d, supp_m, supp_r, q_m, q_r = batch   

        pos_pred, wts = self(d, supp_m, supp_r, q_m)       


        neg_m = torch.randint(
            0,
            self.hparams.n_movies,
            q_m.shape,
            device=q_m.device,
            dtype=torch.long
        )
        neg_pred, _ = self(d, supp_m, supp_r, neg_m)        

        bpr_loss = -torch.log(torch.sigmoid(pos_pred - neg_pred) + 1e-8).mean()
        self.log("train/bpr_loss", bpr_loss, prog_bar=False)

        mse_loss = F.mse_loss(pos_pred, q_r)
        self.log("train/mse_loss", mse_loss, prog_bar=False)

        zero_mask = (supp_m.sum(dim=1) == 0).float()           
        if zero_mask.any():
            demog_wt = wts[:, 2]                              
            aux_zero = ((1.0 - demog_wt) * zero_mask).mean()
        else:
            aux_zero = torch.tensor(0.0, device=pos_pred.device)
        self.log("train/aux_zero", aux_zero, prog_bar=False)

        ent = -(wts * torch.log(wts + 1e-8)).sum(dim=1).mean()
        entropy_weight = 0.1
        self.log("train/gate_entropy", ent, prog_bar=False)

        reg_weight = 1e-5
        l2_reg = 0.0
        for p in self.gate_linear.parameters():
            l2_reg += p.pow(2).sum()
        l2_reg = reg_weight * l2_reg
        self.log("train/l2_gate", l2_reg, prog_bar=False)
        loss = (
              1.0 * bpr_loss
            + 0.5 * mse_loss
            + 0.5 * aux_zero
            - 0.1 * ent
            + l2_reg
        )
        self.log("train/total_loss", loss, prog_bar=True)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr", lr, prog_bar=False)

        return loss
    
    def validation_step(self, batch, _):
        d, s_m, s_r, q_m, q_r = batch
        y_hat, wts = self(d, s_m, s_r, q_m)
        rmse = torch.sqrt(F.mse_loss(y_hat, q_r))
        self.log("val_rmse", rmse, prog_bar=True)
        self.log("temp", self.temperature.item(), prog_bar=False)
        self.log("gate_demog_weight", wts[:,2].mean(), prog_bar=True)


    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-6
        )