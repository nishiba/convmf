# coding: utf-8

import os
import chainer
from typing import List

import numpy as np
import pandas as pd
from chainer import training, iterators, optimizers, serializers
from chainer.training import extensions
from gensim.corpora.dictionary import Dictionary

from convmf.model.convmf import ConvMF, ConvMFUpdater
from convmf.model.matrix_factorization import RatingData


def make_rating_data() -> List[RatingData]:
    ratings = pd.read_csv(os.path.join('data', 'ratings.csv')).rename(columns={'movie': 'item'})
    ratings.user = ratings.user.astype(np.int32)
    ratings.item = ratings.item.astype(np.int32)
    ratings.rating = ratings.rating.astype(np.float32)
    return [RatingData(*t) for t in ratings.itertuples(index=False)]


def make_item_descriptions():
    descriptions = pd.read_csv(os.path.join('data', 'descriptions.csv')).rename(columns={'movie': 'item'})
    texts = descriptions.description
    texts = texts.apply(lambda x: x.strip().split())
    dictionary = Dictionary(texts.values)
    dictionary.filter_extremes()
    eos_id = len(dictionary.keys())

    # to index list
    texts = texts.apply(lambda x: dictionary.doc2idx(x, unknown_word_index=eos_id))
    texts = texts.apply(lambda x: np.array([a for a in x if a != eos_id]))
    max_sentence_length = max(texts.apply(len))

    # padding
    texts = texts.apply(lambda x: np.pad(x, (0, max_sentence_length - len(x)), 'constant', constant_values=(0, eos_id)))

    # change types
    texts = texts.apply(lambda x: x.astype(np.int32))
    descriptions.id = descriptions.id.astype(np.int32)

    return descriptions.id.values, texts.values, len(dictionary.keys()) + 1


def train_model():
    batch_size = 50
    n_epoch = 50
    gpu = 0

    ratings = make_rating_data()
    filter_windows = [3, 4, 5]
    max_sentence_length = 60
    movie_ids, item_descriptions, n_word = make_item_descriptions()
    n_out_channel = 100
    dropout_ratio = 0.5
    n_factor = 300
    user_lambda = 0.001
    item_lambda = 0.001

    model = ConvMF(ratings=ratings,
                   filter_windows=filter_windows,
                   max_sentence_length=max_sentence_length,
                   n_word=n_word,
                   item_descriptions=item_descriptions,
                   n_out_channel=n_out_channel,
                   dropout_ratio=dropout_ratio,
                   n_factor=n_factor,
                   user_lambda=user_lambda,
                   item_lambda=item_lambda)

    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    model.fit_mf(n_trial=3)
    train_iter = iterators.SerialIterator(list(zip(item_descriptions, movie_ids)), batch_size, shuffle=True)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (n_epoch, 'epoch'), out='result')
    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.PrintReport(entries=[
            'epoch',
            'elapsed_time']))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(ConvMFUpdater(model))
    trainer.run()

    model.to_cpu()
    serializers.save_npz('./result/model.npz', model)


if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    train_model()
