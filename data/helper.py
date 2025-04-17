import pandas as pd
import numpy as np
import json
import os
from objs import Genre, genre_map
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

def load_movies() -> pd.DataFrame:
    cfg = load_config()
    ds = cfg["dataset"]
    movies_path = (Path(__file__).resolve().parent.parent / ds["path"] / ds["movies_file"]).resolve()

    return pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=["MovieID", "Title", "Genres"],
        encoding="ISO‑8859‑1",
    )

def load_ratings() -> pd.DataFrame:
    cfg        = load_config()
    ds         = cfg["dataset"]
    ratings_path  = (Path(__file__).resolve().parent.parent / ds["path"] / ds["ratings_file"]).resolve()

    return pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        encoding="ISO-8859-1",
    )

def get_movie_ratings(id, ratings=load_ratings(), exclude_movie_col=True):
    movie_ratings = ratings[ratings['MovieID'] == id]
    if exclude_movie_col:
        return movie_ratings[['UserID', 'Rating']]
    else:
        return movie_ratings[['UserID', 'MovieID', 'Rating']]
    
def get_movie(id, movies=load_movies()):
    movie = movies[movies['MovieID'] == id]
    if not movie.empty:
        return movie
    else:
        return None
    
def parse_genres(genre_string):
    return [
        genre_map[name]
        for name in genre_string.split('|')
        if name in genre_map
    ]