import torch, random
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class ColdStartDataset(Dataset):
    def __init__(self, ratings_df, demog_tensor, k=3):
        self.k = k
        self.demog = demog_tensor           # (n_users, 4)
        # User rating history
        self.user_hist = ratings_df.groupby("uid").apply(
            lambda df: list(zip(df.mid.values, df.Rating.values))
        ).to_dict()
        self.users = list(self.user_hist.keys())

    def __len__(self): return len(self.users)

    def __getitem__(self, idx):
        uid = self.users[idx]
        # All the ratings that the users ever gave, in history
        hist = self.user_hist[uid]
        # Support set for k-shot predictions
        random.shuffle(hist)
        supp = hist[: self.k]          # if k=0 → supp = []
        query = hist[self.k :]              # at least 1 because ML‑1M is dense

        # tuples → separate lists
        supp_m, supp_r = zip(*supp)
        q_m, q_r = zip(*query)
        return (
            uid,
            torch.tensor(self.demog[uid]),
            torch.tensor(supp_m, dtype=torch.long),
            torch.tensor(supp_r, dtype=torch.float),
            torch.tensor(q_m, dtype=torch.long),
            torch.tensor(q_r, dtype=torch.float),
        )


class ML1MDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, demog, batch_size=256, k=3):
        super().__init__()
        self.train = ColdStartDataset(train_df, demog, k)
        self.val   = ColdStartDataset(val_df,   demog, k)
        self.bs = batch_size

    def collate(self, batch):
        # pad support to length k, pad query to length 1 (use first q movie)
        uids, demo, s_m, s_r, q_m, q_r = zip(*batch)
        max_k = max(len(x) for x in s_m)
        pad = lambda lst, fill: [list(x) + [fill]*(max_k-len(x)) for x in lst]
        s_m = torch.tensor(pad(s_m, 0))
        s_r = torch.tensor(pad(s_r, 3.5))
        demo = torch.stack(demo)
        q_m  = torch.tensor([x[0] for x in q_m])
        q_r  = torch.tensor([x[0] for x in q_r])
        return demo, s_m, s_r, q_m, q_r

    def train_dataloader(self):
        return DataLoader(self.train, self.bs, shuffle=True, collate_fn=self.collate)

    def val_dataloader(self):
        return DataLoader(self.val, self.bs, shuffle=False, collate_fn=self.collate)