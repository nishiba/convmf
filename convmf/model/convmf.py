# coding: utf-8

from enum import Enum
from typing import List

import chainer
import chainer.functions as functions
import chainer.links as links
import six
from chainer.dataset import convert

from convmf.model.matrix_factorization import RatingData


class ConvMF(chainer.Chain):
    def __init__(self,
                 ratings: List[RatingData],
                 descriptions,
                 n_factor=300,
                 user_lambda=10,
                 item_lambda=100,
                 use_cnn=True):
        super(ConvMF, self).__init__()
        self.n_factor = n_factor
        self.user_lambda = user_lambda
        self.item_lambda = item_lambda
        self.descriptions = descriptions
        self.use_cnn = use_cnn
        self.convolution_item_factor = self.xp.zeros(shape=(len(descriptions), n_factor), dtype=self.xp.float32)
        n_item = max([x.item for x in ratings]) + 1
        n_user = max([x.user for x in ratings]) + 1

        # model architecture
        with self.init_scope():
            self.user_factor = links.EmbedID(n_user, self.n_factor)
            self.item_factor = links.EmbedID(n_item, self.n_factor)

    def __call__(self, user, item, rating=None):
        user_factor = self.user_factor(user)
        item_factor = self.item_factor(item)

        if chainer.configuration.config.train:
            approximates = functions.matmul(functions.expand_dims(user_factor, axis=1),
                                            functions.expand_dims(item_factor, axis=1),
                                            transb=True)
            loss = functions.mean_squared_error(functions.expand_dims(rating, axis=1), approximates)
            user_weight = functions.mean(functions.square(user_factor))
            loss += self.user_lambda * user_weight
            if self.use_cnn:
                item_error = functions.mean_squared_error(self.convolution_item_factor[item], item_factor)
                loss += self.item_lambda * item_error
            else:
                item_error = functions.mean(functions.square(item_factor))
                loss += self.item_lambda * item_error
            chainer.reporter.report({'loss': loss}, self)
            return loss

        if rating is not None:
            if self.use_cnn:
                approximates = functions.matmul(functions.expand_dims(user_factor, axis=1),
                                                functions.expand_dims(self.convolution_item_factor[item], axis=1),
                                                transb=True)
            else:
                approximates = functions.matmul(functions.expand_dims(user_factor, axis=1),
                                                functions.expand_dims(item_factor, axis=1),
                                                transb=True)
            loss = functions.mean_squared_error(functions.expand_dims(rating, axis=1), approximates)
            chainer.reporter.report({'loss': loss}, self)
            return loss

        if self.use_cnn:
            return functions.matmul(user_factor, self.convolution_item_factor[item], transb=True)
        return functions.matmul(user_factor, item_factor, transb=True)

    def update_convolution_item_factor(self, cnn: 'CNNRand', batch_size=50):
        with chainer.configuration.using_config('train', False):
            for i in range(0, len(self.descriptions), batch_size):
                self.convolution_item_factor[i:i + batch_size] = cnn(x=self.descriptions[i:i + batch_size]).data

    def to_cpu(self):
        super(ConvMF, self).to_cpu()
        from chainer.backends import cuda
        self.descriptions = cuda.to_cpu(self.descriptions)
        self.convolution_item_factor = cuda.to_cpu(self.convolution_item_factor)
        return self

    def to_gpu(self, device=None):
        from chainer.backends import cuda
        with cuda._get_device(device):
            super(ConvMF, self).to_gpu()
            self.descriptions = cuda.to_gpu(self.descriptions, device=device)
            self.convolution_item_factor = cuda.to_gpu(self.convolution_item_factor, device=device)
        return self


class ConvolutionList(chainer.ChainList):
    def __init__(self, n_in_channel: int, n_out_channel: int, n_factor: int, filter_windows: List[int]):
        link_list = [links.Convolution2D(n_in_channel, n_out_channel, (window, n_factor), nobias=False, pad=0) for window in filter_windows]
        super(ConvolutionList, self).__init__(*link_list)


class CNNRand(chainer.Chain):
    def __init__(self, filter_windows: List[int], max_sentence_length, n_word, n_factor, n_out_channel=100, dropout_ratio=0.5):
        super(CNNRand, self).__init__()
        # hyperparameters
        self.filter_windows = filter_windows
        self.max_sentence_length = max_sentence_length
        self.n_word = n_word
        self.n_factor = n_factor
        self.n_in_channel = 1
        self.n_out_channel = n_out_channel
        self.dropout_ratio = dropout_ratio
        self.item_factors = None

        # model architecture
        with self.init_scope():
            self.embedId = links.EmbedID(self.n_word, self.n_factor)
            self.convolution_links = ConvolutionList(n_in_channel=self.n_in_channel,
                                                     n_out_channel=self.n_out_channel,
                                                     n_factor=self.n_factor,
                                                     filter_windows=self.filter_windows)
            self.fully_connected = links.Linear(self.n_out_channel * len(self.filter_windows), self.n_factor)

    def __call__(self, x, t=None):
        # item embedding
        embedding = functions.expand_dims(self.embedId(x), axis=1)
        convolutions = [functions.relu(c(embedding)) for c in self.convolution_links]
        poolings = functions.concat([functions.max_pooling_2d(c, ksize=(c.shape[2])) for c in convolutions], axis=2)
        y = functions.dropout(self.fully_connected(poolings), ratio=self.dropout_ratio)

        if t is not None:
            loss = functions.mean_squared_error(y, self.item_factors[t])
            chainer.reporter.report({'loss': loss}, self)
            return loss
        else:
            return y

    def update_item_factors(self, item_factors):
        self.item_factors = item_factors


class ConvMFUpdater(chainer.training.Updater):
    def __init__(self, iterators, optimizers, mf: ConvMF, cnn: CNNRand, device=None) -> None:
        self._iterators = iterators
        self._optimizers = optimizers
        self.device = device
        self.iteration = 0
        self.converter = convert.concat_examples
        self.current_name = 'cnn'
        self.mf = mf
        self.cnn = cnn

    @property
    def epoch(self):
        return self._iterators[self.current_name].epoch

    @property
    def epoch_detail(self):
        return self._iterators[self.current_name].epoch_detail

    @property
    def previous_epoch_detail(self):
        return self._iterators[self.current_name].previous_epoch_detail

    @property
    def is_new_epoch(self):
        return self._iterators[self.current_name].is_new_epoch

    def finalize(self):
        """Finalizes the updater object.

        This method calls the `finalize` method of each iterator that
        this updater has.
        It is called at the end of training loops.

        """
        for iterator in six.itervalues(self._iterators):
            iterator.finalize()

    def get_optimizer(self, name):
        """Gets the optimizer of given name.

        Args:
            name (str): Name of the optimizer.

        Returns:
            ~chainer.Optimizer: Corresponding optimizer.

        """
        return self._optimizers[name]

    def get_all_optimizers(self):
        """Gets a dictionary of all optimizers for this updater.

        Returns:
            dict: Dictionary that maps names to optimizers.

        """
        return dict(self._optimizers)

    def get_iterator(self, name):
        """Gets the dataset iterator of given name.

        Args:
            name (str): Name of the dataset iterator.

        Returns:
            ~chainer.dataset.Iterator: Corresponding dataset iterator.

        """
        return self._iterators[name]

    def update(self):
        """Updates the parameters of the target model.

        This method implements an update formula for the training task,
        including data loading, forward/backward computations, and actual
        updates of parameters.

        This method is called once at each iteration of the training loop.

        """
        self.update_core()
        self.iteration += 1

    def update_core(self):
        if self.is_new_epoch:
            self._switch()
        # update
        self._update_core(self.current_name)

    def _update_core(self, name):
        batch = self._iterators[name].next()
        in_arrays = self.converter(batch, self.device)

        optimizer = self._optimizers[name]
        loss_func = optimizer.target

        if isinstance(in_arrays, tuple):
            optimizer.update(loss_func, *in_arrays)
        elif isinstance(in_arrays, dict):
            optimizer.update(loss_func, **in_arrays)
        else:
            optimizer.update(loss_func, in_arrays)

    def _switch(self):
        if self.current_name == 'cnn':
            self.current_name = 'mf'
            self.mf.update_convolution_item_factor(self.cnn)
        else:
            self.current_name = 'cnn'
            self.cnn.update_item_factors(self.mf.item_factor.W.data)

    def serialize(self, serializer):
        """Serializes the current state of the updater object."""
        for name, iterator in six.iteritems(self._iterators):
            iterator.serialize(serializer['iterator:' + name])

        for name, optimizer in six.iteritems(self._optimizers):
            optimizer.serialize(serializer['optimizer:' + name])
            optimizer.target.serialize(serializer['model:' + name])

        self.iteration = serializer('iteration', self.iteration)
