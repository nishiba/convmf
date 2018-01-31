# coding: utf-8
from typing import List

import chainer
import chainer.functions as functions
import chainer.links as links

try:
    from chainer.backends import cuda
except ImportError:
    pass
import numpy as np
from chainer.training import extension, make_extension
from cnn_sc.model.cnn_rand import CNNRand, TaskType

from convmf.model.matrix_factorization import RatingData, MatrixFactorization


@make_extension(trigger=(1, 'epoch'))
class ConvMFUpdater(extension.Extension):
    def __init__(self, model):
        self.model = model

    def __call__(self, trainer=None):
        self.model.update_latent_factor()


class ConvMF(chainer.Chain):
    def __init__(self,
                 ratings: List[RatingData],
                 filter_windows: List[int],
                 max_sentence_length: int,
                 item_descriptions: List[np.ndarray],
                 n_word,
                 n_out_channel=100,
                 dropout_ratio=0.5,
                 n_factor=300,
                 user_lambda=0.001,
                 item_lambda=0.001,
                 mf: MatrixFactorization = None):
        super(ConvMF, self).__init__()
        self.n_factor = n_factor
        self.item_descriptions = item_descriptions
        n_item = max(max([x.item for x in ratings]), len(item_descriptions))  # TODO(nishiba) fix.
        self.mf = mf
        if self.mf is None:
            self.mf = MatrixFactorization(ratings=ratings, n_factor=n_factor, user_lambda=user_lambda, item_lambda=item_lambda, n_item=n_item)

        # model architecture
        with self.init_scope():
            self.convolution = CNNRand(filter_windows=filter_windows, max_sentence_length=max_sentence_length, n_word=n_word, n_factor=n_factor,
                                       n_out_channel=n_out_channel, n_class=n_factor, dropout_ratio=dropout_ratio, mode=TaskType.Embedding)

    def __call__(self, x, y=None, train=True):
        if train:
            loss = self.convolution(x=x, t=y, train=True)
            chainer.reporter.report({'loss': loss}, self)
            return loss
        return self.convolution(x=x, train=False)

    def predict(self, users: List[int], items: List[int]) -> List[float]:
        item_factors = self.convolution(x=np.array([self.item_descriptions[i] for i in items]), train=False)
        user_factors = [self.mf.user_factors[:, u] for u in users]
        predictions = [self.xp.inner(u, i.data) for u, i in zip(user_factors, item_factors)]
        return predictions

    def get_item_factors(self, items: List[int]) -> List[np.ndarray]:
        item_factors = self.convolution(x=np.array([self.item_descriptions[i] for i in items]), train=False)
        return [i.data for i in item_factors]

    def fit_mf(self, n_trial=3):
        self.mf.fit(n_trial=n_trial)

    def update_latent_factor(self, n_trial=3):
        # TODO(nishiba) must make 'train' off
        self.mf.fit(n_trial=n_trial, additional=self._embedding())
        # TODO(nishiba) must make 'train' on

    def _embedding(self):
        return [self.convolution(x=np.array([d]), train=False)[0].data for d in self.item_descriptions]
