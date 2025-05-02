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
            self.demog[uid].clone().detach(),
            torch.tensor(supp_m, dtype=torch.long),
            torch.tensor(supp_r, dtype=torch.float),
            torch.tensor(q_m, dtype=torch.long),
            torch.tensor(q_r, dtype=torch.float),
        )


class ML1MDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df, val_df, demog,
        batch_size=256,
        k_range=(0,3,5,10),
        k_max: int = 10,
    ):
        super().__init__()
        # we no longer pass a fixed k into the dataset—support‐set size is chosen in collate
        self.train   = ColdStartDataset(train_df, demog)
        self.val     = ColdStartDataset(val_df,   demog)
        self.bs      = batch_size
        self.k_range = k_range  # e.g. (0,3,5,10)
        self.k_max   = k_max    # always pad to this length

    def collate(self, batch):
        uids, demo, s_m, s_r, q_m, q_r = zip(*batch)

        # 1) pick k' for each example
        if self._is_training:
            lo, hi = min(self.k_range), max(self.k_range)
            # sample uniformly from {lo, lo+1, …, hi}
            k_prime = torch.randint(lo, hi + 1, (len(batch),))
        else:
            # always use the maximum support size at validation
            k_prime = torch.full(
                (len(batch),),
                fill_value=self.k_max,
                dtype=torch.long,
            )

        # 2) trim each support‐set down to k_prime[i]
        new_s_m, new_s_r = [], []
        for i, (movies, ratings) in enumerate(zip(s_m, s_r)):
            kp = k_prime[i].item()
            new_s_m.append(movies[:kp])
            new_s_r.append(ratings[:kp])

        # 3) pad *all* to self.k_max (not dynamic)
        pad_to = self.k_max
        def pad(list_of_lists, fill_value):
            return [list(x) + [fill_value] * (pad_to - len(x)) for x in list_of_lists]

        s_m = torch.tensor(pad(new_s_m, 0), dtype=torch.long)
        s_r = torch.tensor(pad(new_s_r, 3.5), dtype=torch.float)

        # 4) the rest stays the same
        demo = torch.stack(demo)
        q_m  = torch.tensor([x[0] for x in q_m], dtype=torch.long)
        q_r  = torch.tensor([x[0] for x in q_r], dtype=torch.float)
        return demo, s_m, s_r, q_m, q_r

    def train_dataloader(self):
        self._is_training = True
        return DataLoader(self.train, self.bs, shuffle=True, collate_fn=self.collate)

    def val_dataloader(self):
        self._is_training = False
        return DataLoader(self.val,   self.bs, shuffle=False, collate_fn=self.collate)