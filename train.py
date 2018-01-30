# coding: utf-8

import os
import pickle
from typing import List

import chainer
import numpy as np
import pandas as pd
from chainer import training, iterators, optimizers, serializers
from chainer.training import extensions
from gensim.corpora.dictionary import Dictionary
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from convmf.model.convmf import ConvMF, ConvMFUpdater
from convmf.model.matrix_factorization import RatingData, MatrixFactorization


def make_rating_data() -> List[RatingData]:
    ratings = pd.read_csv(os.path.join('data', 'ratings.csv')).rename(columns={'movie': 'item'})
    ratings.user = ratings.user.astype(np.int32)
    ratings.item = ratings.item.astype(np.int32)
    ratings.rating = ratings.rating.astype(np.float32)
    return [RatingData(*t) for t in ratings.itertuples(index=False)]


def make_item_descriptions(max_sentence_length=None):
    descriptions = pd.read_csv(os.path.join('data', 'descriptions.csv')).rename(columns={'movie': 'item'})
    texts = descriptions.description
    texts = texts.apply(lambda x: x.strip().split())
    dictionary = Dictionary(texts.values)
    dictionary.filter_extremes()
    eos_id = len(dictionary.keys())

    # to index list
    texts = texts.apply(lambda x: dictionary.doc2idx(x, unknown_word_index=eos_id))
    texts = texts.apply(lambda x: np.array([a for a in x if a != eos_id]))
    max_sentence_length = max(texts.apply(len)) if max_sentence_length is None else min(max(texts.apply(len)), max_sentence_length)

    # padding
    texts = texts.apply(lambda x: x[:max_sentence_length])
    texts = texts.apply(lambda x: np.pad(x, (0, max_sentence_length - len(x)), 'constant', constant_values=(0, eos_id)))

    # change types
    texts = texts.apply(lambda x: x.astype(np.int32))
    descriptions.id = descriptions.id.astype(np.int32)

    return descriptions.id.values, texts.values, len(dictionary.keys()) + 1


def train_convmf():
    batch_size = 50
    n_epoch = 50
    n_sub_epoch = 10
    gpu = 0

    ratings = make_rating_data()
    filter_windows = [3, 4, 5]
    max_sentence_length = 300
    movie_ids, item_descriptions, n_word = make_item_descriptions(max_sentence_length=max_sentence_length)
    n_out_channel = 2
    n_factor = 300
    dropout_ratio = 0.5
    user_lambda = 10
    item_lambda = 100

    with open('./result/mf.pkl', 'rb') as f:
        mf = pickle.load(f)

    model = ConvMF(ratings=ratings,
                   filter_windows=filter_windows,
                   max_sentence_length=max_sentence_length,
                   n_word=n_word,
                   item_descriptions=item_descriptions,
                   n_out_channel=n_out_channel,
                   dropout_ratio=dropout_ratio,
                   n_factor=n_factor,
                   user_lambda=user_lambda,
                   item_lambda=item_lambda,
                   mf=mf)

    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    train_ratings, test_ratings = train_test_split(ratings, test_size=0.1, random_state=123)

    train_iter = iterators.SerialIterator(list(zip(item_descriptions, movie_ids)), batch_size, shuffle=True)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (n_sub_epoch, 'epoch'), out='result')
    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.PrintReport(entries=[
            'epoch',
            'main/loss',
            'test/main/loss',
            'elapsed_time']))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(ConvMFUpdater(model))

    for n in range(0, n_epoch, n_sub_epoch):
        trainer.run()
        with chainer.using_config('train', False):
            model.to_cpu()
            predict = model.predict(users=[r.user for r in test_ratings], items=[r.item for r in test_ratings])
            rmse = np.sqrt(np.mean(np.square(predict - np.array([r.rating for r in test_ratings]))))
            print('rmse: %.4f' % rmse)
        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()  # Make a specified GPU current
            model.to_gpu()  # Copy the model to the GPU

    model.to_cpu()
    serializers.save_npz('./result/convmf.npz', model)


def make_negative_test_case(ratings: List[RatingData], size: int) -> List[RatingData]:
    users = [r.user for r in ratings]
    items = [r.item for r in ratings]
    positive_cases = list(zip(users, items))
    sample_size = size + len(ratings)
    samplings = list(zip(np.random.choice(users, size=sample_size), np.random.choice(items, size=sample_size)))
    negative_cases = shuffle(list(set(samplings) - set(positive_cases)))[:size]
    return [RatingData(user=u, item=i, rating=0) for u, i in negative_cases]


def train_mf():
    ratings = make_rating_data()
    n_factor = 300
    n_trial = 10
    user_lambda = 10
    item_lambda = 100

    n_item = len(pd.DataFrame(ratings).item.unique())
    train, test = train_test_split(ratings, test_size=0.1, random_state=123)
    model = MatrixFactorization(ratings=train, n_factor=n_factor, user_lambda=user_lambda, item_lambda=item_lambda, n_item=n_item)
    for n in range(n_trial):
        model.fit(n_trial=1)
        predict = model.predict(users=[r.user for r in test], items=[r.item for r in test])
        rmse = np.sqrt(np.mean(np.square(predict - np.array([r.rating for r in test]))))
        print('rmse: %.4f' % rmse)

    with open('./result/mf.pkl', 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    train_convmf()
