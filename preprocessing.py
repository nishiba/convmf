# coding: utf-8
import os
import re
import pandas as pd
import numpy as np
import json


def make_movie_data():
    credit_df = pd.read_csv(os.path.join('data', 'tmdb-5000-movie-dataset', 'tmdb_5000_credits.csv'))[['movie_id', 'cast', 'crew']]
    credit_df['cast'] = credit_df['cast'].apply(json.loads)
    credit_df['cast'] = credit_df['cast'].apply(lambda xs: [x['name'] for x in xs[:10]])
    credit_df['crew'] = credit_df['crew'].apply(json.loads)
    credit_df['crew'] = credit_df['crew'].apply(lambda xs: [x['name'] for x in xs[:10]])

    movie_df = pd.read_csv(os.path.join('data', 'tmdb-5000-movie-dataset', 'tmdb_5000_movies.csv'))[['id', 'title', 'overview']]
    movie_df.title = movie_df.title.apply(lambda x: x.lower())
    merged = pd.merge(movie_df, credit_df, left_on='id', right_on='movie_id')
    return merged[['title', 'overview', 'cast', 'crew']]


def make_descriptions():
    title_id = pd.read_csv(os.path.join('data', 'ml-10M100K', 'movies.dat'), sep='::', engine='python', names=['id', 'title', 'tag'])
    title_id = title_id[['id', 'title']]
    title_id.title = title_id.title.apply(lambda x: re.sub(r'\(\d+\)', '', x).rstrip())
    title_id.title = title_id.title.apply(lambda x: x.lower())

    movie_df = make_movie_data()

    merged = pd.merge(movie_df, title_id, on='title')
    merged = merged[['id', 'overview', 'cast', 'crew']].copy()
    merged = merged.rename(columns={'overview': 'description'})
    merged.id = merged.id.astype(np.int32)
    return merged


def make_ratings():
    ratings = pd.read_csv(os.path.join('data', 'ml-10M100K', 'ratings.dat'), sep='::', engine='python', names=['user', 'movie', 'rating', 'timestamp'])
    ratings = ratings[['user', 'movie', 'rating']].copy()
    ratings.user = ratings.user.astype(np.int32)
    ratings.movie = ratings.movie.astype(np.int32)
    ratings.rating = ratings.rating.astype(np.float32)
    return ratings


def preprocess():
    ratings = make_ratings()
    descriptions = make_descriptions()

    # re-indexing
    users = ratings.user.unique()
    user_map = dict(zip(users, range(len(users))))
    movies = descriptions.id.unique()
    movie_map = dict(zip(movies, range(len(movies))))

    ratings.user = ratings.user.apply(lambda x: user_map.get(x, None))
    ratings.movie = ratings.movie.apply(lambda x: movie_map.get(x, None))
    descriptions.id = descriptions.id.apply(lambda x: movie_map.get(x, None))
    ratings = ratings.dropna()
    descriptions = descriptions.dropna()
    ratings.to_csv(os.path.join('data', 'ratings.csv'), index=False)
    descriptions.to_csv(os.path.join('data', 'descriptions.csv'), index=False)


if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    preprocess()
