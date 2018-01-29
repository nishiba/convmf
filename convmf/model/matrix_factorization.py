# coding: utf-8
from typing import NamedTuple, List
from typing import Optional

import numpy as np
import pandas as pd

import chainer
import cupy as cp

class RatingData(NamedTuple):
    user: int
    item: int
    rating: float


class IndexRatingSet(NamedTuple):
    indices: List[int]
    ratings: List[float]

    @staticmethod
    def to_gpu(x):
        return IndexRatingSet(indices=chainer.cuda.to_gpu(x.indices), ratings=chainer.cuda.to_gpu(x.ratings))

    @staticmethod
    def to_cpu(x):
        return IndexRatingSet(indices=chainer.cuda.to_cpu(x.indices), ratings=chainer.cuda.to_cpu(x.ratings))


class MatrixFactorization(object):
    def __init__(self, ratings: List[RatingData], n_factor=300, user_lambda=0.001, item_lambda=0.001, n_item: int = None):
        data = pd.DataFrame(ratings)
        self.xp = np
        self.n_user = max(data.user.unique()) + 1
        self.n_item = n_item if n_item is not None else max(data.item.unique()) + 1
        self.n_factor = n_factor
        self.user_lambda = user_lambda
        self.item_lambda = item_lambda
        self.user_factors = self.xp.random.normal(size=(self.n_factor, self.n_user)).astype(np.float32)
        self.item_factors = self.xp.random.normal(size=(self.n_factor, self.n_item)).astype(np.float32)
        self.user_item_list = {i: v for i, v in data.groupby('user').apply(lambda x: IndexRatingSet(indices=x.item.values, ratings=x.rating.values)).items()}
        self.item_user_list = {i: v for i, v in data.groupby('item').apply(lambda x: IndexRatingSet(indices=x.user.values, ratings=x.rating.values)).items()}

    def fit(self, n_trial=5, additional: Optional[List[np.ndarray]] = None):
        for n in range(n_trial):
            print('%d/%d' % (n, n_trial))
            self.update_user_factors()
            self.update_item_factors(additional)

    def update_user_factors(self):
        for u in self.user_item_list.keys():
            indices = self.user_item_list[u].indices
            ratings = self.user_item_list[u].ratings
            v = self.item_factors[:, indices]
            a = self.xp.dot(v, v.T)
            a[np.diag_indices_from(a)] += self.user_lambda
            b = self.xp.dot(v, ratings)
            self.user_factors[:, u] = self.xp.linalg.solve(a, b)

    def update_item_factors(self, additional: Optional[List[np.ndarray]] = None):
        for v in self.item_user_list.keys():
            indices = self.item_user_list[v].indices
            ratings = self.item_user_list[v].ratings
            u = self.user_factors[:, indices]
            a = self.xp.dot(u, u.T)
            a[np.diag_indices_from(a)] += self.item_lambda
            b = self.xp.dot(u, ratings)
            if additional is not None:
                b += self.item_lambda * additional[v]
            self.item_factors[:, v] = self.xp.linalg.solve(a, b)

    def to_gpu(self):
        self.user_factors = chainer.cuda.to_gpu(self.user_factors)
        self.item_factors = chainer.cuda.to_gpu(self.item_factors)
        self.user_item_list = {k: IndexRatingSet.to_gpu(v) for k, v in self.user_item_list.items()}
        self.item_user_list = {k: IndexRatingSet.to_gpu(v) for k, v in self.item_user_list.items()}
        self.xp = cp

    def to_cpu(self):
        self.user_factors = chainer.cuda.to_cpu(self.user_factors)
        self.item_factors = chainer.cuda.to_cpu(self.item_factors)
        self.user_item_list = {k: IndexRatingSet.to_cpu(v) for k, v in self.user_item_list.items()}
        self.item_user_list = {k: IndexRatingSet.to_cpu(v) for k, v in self.item_user_list.items()}
        self.xp = np
