import pandas as pd
import numpy as np
import json
import torch
import os
from pathlib import Path
from data.datasets import ML1MDataModule

def load_config(config_path = None, verbose=False) -> dict:
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.json"
    else:
        config_path = Path(config_path).expanduser().resolve()
    if verbose:
        print("Loading configuration file:", config_path)
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found.")
        raise FileNotFoundError
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the configuration file {config_path}.")
        raise json.JSONDecodeError

def load_movies(config_path = None, verbose=False) -> pd.DataFrame:
    cfg = load_config(config_path=config_path, verbose=verbose)
    ds = cfg["dataset"]
    movies_path = (Path(__file__).resolve().parent.parent / ds["path"] / ds["movies_file"]).resolve()

    return pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=["MovieID", "Title", "Genres"],
        encoding="ISO‑8859‑1",
    )

def load_ratings(config_path = None, verbose=False) -> pd.DataFrame:
    cfg        = load_config(config_path=config_path, verbose=verbose)
    ds         = cfg["dataset"]
    ratings_path  = (Path(__file__).resolve().parent.parent / ds["path"] / ds["ratings_file"]).resolve()

    return pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        encoding="ISO-8859-1",
    )

def load_users(config_path = None, verbose=False) -> pd.DataFrame:
    cfg = load_config(config_path=config_path, verbose=verbose)
    ds = cfg["dataset"]
    users_path = (Path(__file__).resolve().parent.parent / ds["path"] / ds["users_file"]).resolve()

    return pd.read_csv(
        users_path,
        sep="::",
        engine="python",
        names=["UserID","Gender","Age","Occupation","Zipcode"],
        encoding="ISO‑8859‑1",
    )


def gen_demographic_table(config_path = None, verbose=False):
    movies = load_movies(config_path=config_path, verbose=verbose)
    ratings = load_ratings(config_path=config_path, verbose=verbose)
    users = load_users(config_path=config_path, verbose=verbose)

    # Remap user ids and movie ids to be continuous from 0 to n-1
    uid_map = {u: i for i, u in enumerate(users.UserID.unique())}
    mid_map = {m: i for i, m in enumerate(ratings.MovieID.unique())}

    ratings["uid"] = ratings.UserID.map(uid_map)
    ratings["mid"] = ratings.MovieID.map(mid_map)
    users["uid"] = users.UserID.map(uid_map)

    # Sort users just in case
    users_sorted = users.sort_values("uid")

    gender_id = users_sorted.Gender.map({"M": 0, "F": 1}).to_numpy(dtype=np.int64)
    age_id    = users_sorted.Age.to_numpy(dtype=np.int64)
    occ_id    = users_sorted.Occupation.to_numpy(dtype=np.int64)
    zip_id    = users_sorted.Zipcode.str[:2].astype(int).to_numpy(dtype=np.int64)

    # fast, contiguous (n_users, 4) array
    demog_np = np.column_stack([gender_id, age_id, occ_id, zip_id])

    # single zero‑copy jump into PyTorch
    demog = torch.from_numpy(demog_np)

    return ratings[["uid", "mid", "Rating", "Timestamp"]], demog, len(uid_map), len(mid_map)


def load_save_neumf_table(config_path = None, verbose=False):
    cfg = load_config(config_path=config_path, verbose=verbose)
    ckpt_path = (Path(__file__).resolve().parent.parent / cfg["neumf_ckpt"]).resolve()
    assert os.path.exists(ckpt_path), f"Checkpoint file {ckpt_path} does not exist."
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt if isinstance(ckpt, dict) else ckpt["model"]
    assert "embedding_item_mf.weight" in state.keys() and "embedding_item_mlp.weight" in state.keys(), "Invalid checkpoint keys. Check the keys with state.keys()"
    item_gmf = state["embedding_item_mf.weight"].clone()   # shape (n_items, d)
    item_mlp = state["embedding_item_mlp.weight"].clone()  # same shape


    neumf_table_path =(Path(__file__).resolve().parent.parent / cfg["neumf_table"]).resolve()
    torch.save({"item_gmf": item_gmf,
                "item_mlp": item_mlp}, neumf_table_path)
    if verbose:
        print("Saved GMF and MLP tables to", neumf_table_path)
    return item_gmf, item_mlp


def gen_datamodule(k_list, max_k: int, test_size=0.2, ratings = load_ratings(), random_state=42, verbose=False):
    # k-shot split
    if verbose:
        print("Loading data and demographic table...")
    ratings, demog, n_users, n_movies = gen_demographic_table(verbose=verbose)
    user_ids = ratings.uid.unique()                         # array of ~6 000 uids
    n_val = int(len(user_ids) * test_size)
    val_uids = set(
        pd.Series(user_ids).sample(n=n_val, random_state=random_state)
    )
    train_df = ratings[~ratings.uid.isin(val_uids)]
    val_df   = ratings[ratings.uid.isin(val_uids)]
    if verbose:
        print(f"Train set size: {len(train_df)}, Validation set size: {len(val_df)}")
    dm = ML1MDataModule(
        train_df,
        val_df,
        demog,
        batch_size=256,
        k_range=tuple(k_list),   # e.g. (0,3,5,10)
        k_max=max_k,             # e.g. 10
    )
    if verbose:
        print("Data module created.")
    return dm, n_users, n_movies, demog

