import pandas as pd
import numpy as np
import json
import os
from objs import Genre, genre_map

def load_config(config_path = "config.json"):
    """
    Load the configuration file.
    """
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

def load_movies():
    dataset_config = load_config()["dataset"]
    movies = pd.read_csv(
        os.path.join(dataset_config["path"], dataset_config["movies_file"]),
        sep='::',
        engine='python',
        names=['MovieID', 'Title', 'Genres'],
        encoding='ISO-8859-1'
    )
    return movies

def load_ratings():
    dataset_config = load_config()["dataset"]
    ratings = pd.read_csv(
        os.path.join(dataset_config["path"], dataset_config["ratings_file"]),
        sep='::',
        engine='python',
        names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
        encoding='ISO-8859-1'
    )
    return ratings

def get_movie_ratings(id, ratings=load_ratings(), exclude_movie_col=True):
    movie_ratings = ratings[ratings['MovieID'] == id]
    if exclude_movie_col:
        return movie_ratings[['UserID', 'Rating']]
    else:
        return movie_ratings[['UserID', 'MovieID', 'Rating']]
    
def get_movie_title(id, movies=load_movies()):
    movie_title = movies[movies['MovieID'] == id]['Title']
    if not movie_title.empty:
        return movie_title.values[0]
    else:
        return None
    
def parse_genres(genre_string):
    return [
        genre_map[name]
        for name in genre_string.split('|')
        if name in genre_map
    ]