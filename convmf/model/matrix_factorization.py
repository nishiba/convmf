# coding: utf-8
from typing import NamedTuple, List
from typing import Optional

import chainer
import numpy as np
import pandas as pd


class RatingData(NamedTuple):
    user: int
    item: int
    rating: float


class IndexRatingSet(NamedTuple):
    indices: List[int]
    ratings: List[float]


class MatrixFactorization(object):
    def __init__(self, ratings: List[RatingData], n_factor=300, user_lambda=0.001, item_lambda=0.001, n_item: int = None):
        data = pd.DataFrame(ratings)
        self.n_user = max(data.user.unique()) + 1
        self.n_item = n_item if n_item is not None else max(data.item.unique()) + 1
        self.n_factor = n_factor
        self.user_lambda = user_lambda
        self.item_lambda = item_lambda
        self.user_factors = np.random.normal(size=(self.n_factor, self.n_user)).astype(np.float32)
        self.item_factors = np.random.normal(size=(self.n_factor, self.n_item)).astype(np.float32)
        self.user_item_list = {i: v for i, v in data.groupby('user').apply(lambda x: IndexRatingSet(indices=x.item.values, ratings=x.rating.values)).items()}
        self.item_user_list = {i: v for i, v in data.groupby('item').apply(lambda x: IndexRatingSet(indices=x.user.values, ratings=x.rating.values)).items()}

    def fit(self, n_trial=5, additional: Optional[List[np.ndarray]] = None):
        for n in range(n_trial):
            self.update_user_factors()
            self.update_item_factors(additional)

    def predict(self, users: List[int], items: List[int]):
        return np.array([np.inner(self.user_factors[:, u], self.item_factors[:, i]) for u, i in zip(users, items)])

    def update_user_factors(self):
        for u in self.user_item_list.keys():
            indices = self.user_item_list[u].indices
            ratings = self.user_item_list[u].ratings
            v = self.item_factors[:, indices]
            a = np.dot(v, v.T)
            a[np.diag_indices_from(a)] += self.user_lambda
            b = np.dot(v, ratings)
            self.user_factors[:, u] = np.linalg.solve(a, b)

    def update_item_factors(self, additional: Optional[List[np.ndarray]] = None):
        for v in self.item_user_list.keys():
            indices = self.item_user_list[v].indices
            ratings = self.item_user_list[v].ratings
            u = self.user_factors[:, indices]
            a = np.dot(u, u.T)
            a[np.diag_indices_from(a)] += self.item_lambda
            b = np.dot(u, ratings)
            if additional is not None:
                b += self.item_lambda * additional[v]
            self.item_factors[:, v] = np.linalg.solve(a, b)
