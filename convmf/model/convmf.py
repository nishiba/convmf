# coding: utf-8
from typing import List

import chainer
import chainer.functions as functions
import chainer.links as links
from chainer.backends import cuda
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
                 item_lambda=0.001):
        super(ConvMF, self).__init__()
        self.n_factor = n_factor
        self.item_descriptions = item_descriptions
        n_item = max(max([x.item for x in ratings]), len(item_descriptions))  # TODO(nishiba) fix.
        self.mf = MatrixFactorization(ratings=ratings, n_factor=n_factor, user_lambda=user_lambda, item_lambda=item_lambda, n_item=n_item)

        # model architecture
        with self.init_scope():
            self.convolution = CNNRand(filter_windows=filter_windows, max_sentence_length=max_sentence_length, n_word=n_word, n_factor=n_factor,
                                       n_out_channel=n_out_channel, n_class=n_factor, dropout_ratio=dropout_ratio, mode=TaskType.Embedding)

    def __call__(self, x, y=None, train=True):
        if train:
            return self.convolution(x=x, t=self.mf.item_factors[:, y].T, train=True)
        return self.convolution(x=x, train=False)

    def fit_mf(self, n_trial=3):
        self.mf.fit(n_trial=n_trial)

    def update_latent_factor(self, n_trial=3):
        self.mf.fit(n_trial=n_trial, additional=self._embedding())

    def _embedding(self):
        return [self.convolution(x=d, train=False) for d in self.item_descriptions]

    def to_cpu(self):
        super(ConvMF, self).to_cpu()
        self.mf.to_cpu()
        return self

    def to_gpu(self, device=None):
        with cuda._get_device(device):
            super(ConvMF, self).to_gpu()
            self.mf.to_gpu()
        return self
