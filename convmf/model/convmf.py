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
                 n_word,
                 n_out_channel=100,
                 dropout_ratio=0.5,
                 n_factor=300,
                 user_lambda=10,
                 item_lambda=100):
        super(ConvMF, self).__init__()
        self.n_factor = n_factor
        self.user_lambda = user_lambda
        self.item_lambda = item_lambda
        n_item = max([x.item for x in ratings]) + 1
        n_user = max([x.user for x in ratings]) + 1

        # model architecture
        with self.init_scope():
            self.user_factor = links.EmbedID(n_user, self.n_factor)
            self.item_factor = links.EmbedID(n_item, self.n_factor)
            self.convolution = CNNRand(filter_windows=filter_windows, max_sentence_length=max_sentence_length, n_word=n_word, n_factor=n_factor,
                                       n_out_channel=n_out_channel, n_class=n_factor, dropout_ratio=dropout_ratio, mode=TaskType.Embedding)

    def __call__(self, user, description, item=None, ratings=None):
        user_factor = self.user_factor(user)

        if chainer.config.train:
            item_factor = self.item_factor(item)
            approximates = functions.matmul(functions.expand_dims(user_factor, axis=1),
                                            functions.expand_dims(item_factor, axis=1),
                                            transb=True)
            error = functions.mean_squared_error(functions.expand_dims(ratings, axis=1), approximates)
            user_weight = functions.sum(functions.square(user_factor))

            item_error = self.convolution(x=description, t=item_factor, train=True)
            loss = error + self.user_lambda * user_weight + self.item_lambda * item_error
            chainer.reporter.report({'loss': loss}, self)
            return loss

        convolution_item_factor = self.convolution(x=description, train=False)
        return functions.matmul(user_factor, convolution_item_factor, transb=True)

