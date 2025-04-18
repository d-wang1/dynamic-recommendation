import pandas as pd
import numpy as np
import json
import torch
import os
from data.objs import Genre, genre_map
from pathlib import Path

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

# def get_movie_ratings(id, ratings=load_ratings(), exclude_movie_col=True):
#     movie_ratings = ratings[ratings['MovieID'] == id]
#     if exclude_movie_col:
#         return movie_ratings[['UserID', 'Rating']]
#     else:
#         return movie_ratings[['UserID', 'MovieID', 'Rating']]
    
# def get_movie(id, movies=load_movies()):
#     movie = movies[movies['MovieID'] == id]
#     if not movie.empty:
#         return movie
#     else:
#         return None
    
# def parse_genres(genre_string):
#     return [
#         genre_map[name]
#         for name in genre_string.split('|')
#         if name in genre_map
#     ]


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


