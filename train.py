# coding: utf-8
import argparse
import os
from typing import List
import ast

import chainer
import numpy as np
import pandas as pd
from chainer import training, iterators, serializers
from chainer.training import extensions
from gensim.corpora.dictionary import Dictionary
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from convmf.model.convmf import ConvMF, CNNRand
from convmf.model.matrix_factorization import RatingData


def make_rating_data(size: int = None) -> List[RatingData]:
    ratings = pd.read_csv(os.path.join('data', 'ratings.csv')).rename(columns={'movie': 'item'})
    ratings.user = ratings.user.astype(np.int32)
    ratings.item = ratings.item.astype(np.int32)
    ratings.rating = ratings.rating.astype(np.float32)
    if size is not None:
        ratings = ratings.head(size)
    return [RatingData(*t) for t in ratings.itertuples(index=False)]


def make_item_descriptions(max_sentence_length=100):
    descriptions = pd.read_csv(os.path.join('data', 'descriptions.csv')).rename(columns={'movie': 'item'})
    descriptions['crew'] = descriptions['crew'].apply(lambda x: ast.literal_eval(x))
    descriptions['cast'] = descriptions['cast'].apply(lambda x: ast.literal_eval(x))
    texts = descriptions.description
    texts = texts.apply(lambda x: x.strip().split())
    descriptions['description'] = [['crew'] + crew + ['cast'] + cast + ['text'] + text
                                   for crew, cast, text in zip(descriptions.crew, descriptions.cast, texts)]
    texts = descriptions.description
    dictionary = Dictionary(texts.values)
    eos_id = len(dictionary.keys())

    # to index list
    texts = texts.apply(lambda x: dictionary.doc2idx(x, unknown_word_index=eos_id))
    texts = texts.apply(lambda x: np.array([a for a in x if a != eos_id]))
    max_sentence_length = max(texts.apply(len)) if max_sentence_length is None else min(max(texts.apply(len)), max_sentence_length)

    # padding
    texts = texts.apply(lambda x: x[:max_sentence_length])
    texts = texts.apply(lambda x: np.pad(x, (0, max_sentence_length - len(x)), 'constant', constant_values=(0, eos_id)))

    # change types
    texts = texts.apply(lambda x: x.tolist())
    descriptions.id = descriptions.id.astype(np.int32)

    return np.array(descriptions.id.values), np.array(list(texts.values), dtype=np.int32), len(dictionary.keys()) + 1


def make_mf_data(ratings):
    users = np.array([rating.user for rating in ratings], dtype=np.int32)
    items = np.array([rating.item for rating in ratings], dtype=np.int32)
    rates = np.array([rating.rating for rating in ratings], dtype=np.float32).reshape((-1, 1))
    return list(zip(users, items, rates))


def make_cnn_data(ratings, item_descriptions):
    items = np.array(list(set([rating.item for rating in ratings])), dtype=np.int32)
    descriptions = np.array([item_descriptions[i] for i in items], dtype=np.int32)
    return list(zip(descriptions, items))


def train_convmf(mf_batch_size: int, cnn_batch_size: int, n_epoch: int, gpu: int, n_out_channel: int,
                 user_lambda: float, item_lambda: float, n_factor: int):
    ratings = make_rating_data()
    filter_windows = [3, 4, 5]
    max_sentence_length = 200
    movie_ids, item_descriptions, n_word = make_item_descriptions(max_sentence_length=max_sentence_length)
    dropout_ratio = 0.2

    mf = ConvMF(ratings=ratings,
                n_factor=n_factor,
                user_lambda=user_lambda,
                item_lambda=item_lambda,
                descriptions=item_descriptions,
                use_cnn=False)

    cnn = CNNRand(filter_windows=filter_windows,
                  n_word=n_word,
                  n_out_channel=n_out_channel,
                  dropout_ratio=dropout_ratio,
                  n_factor=n_factor)

    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()  # Make a specified GPU current
        mf.to_gpu()  # Copy the model to the GPU
        cnn.to_gpu()  # Copy the model to the GPU

    train_mf, test_mf = train_test_split(make_mf_data(ratings), test_size=0.1, random_state=123)
    train_cnn, test_cnn = train_test_split(make_cnn_data(ratings, item_descriptions), test_size=0.1, random_state=123)
    optimizers = {'mf': chainer.optimizers.Adam(), 'cnn': chainer.optimizers.Adam()}
    optimizers['mf'].setup(mf)
    optimizers['cnn'].setup(cnn)

    train_iter = {'mf': iterators.SerialIterator(train_mf, mf_batch_size, shuffle=True),
                  'cnn': iterators.SerialIterator(train_cnn, cnn_batch_size, shuffle=True)}
    test_iter = {'mf': iterators.SerialIterator(test_mf, mf_batch_size, repeat=False),
                 'cnn': iterators.SerialIterator(test_cnn, cnn_batch_size, repeat=False)}

    # pre-train mf
    def _train_mf():
        updater = training.StandardUpdater(train_iter['mf'], optimizers['mf'], device=gpu)
        trainer = training.Trainer(updater, (10, 'epoch'), out='result')
        trainer.extend(extensions.Evaluator(test_iter['mf'], mf, device=gpu), name='test')
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(entries=['epoch', 'main/loss', 'test/main/loss', 'elapsed_time']))
        trainer.extend(extensions.ProgressBar())
        trainer.run()
        train_iter['mf'].reset()

    _train_mf()
    mf.use_cnn = True

    # pre-train cnn
    def _train_cnn():
        updater = training.StandardUpdater(train_iter['cnn'], optimizers['cnn'], device=gpu)
        trainer = training.Trainer(updater, (50, 'epoch'), out='result')
        trainer.extend(extensions.Evaluator(test_iter['cnn'], cnn, device=gpu), name='test')
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(entries=['epoch', 'main/loss', 'test/main/loss', 'elapsed_time']))
        trainer.extend(extensions.ProgressBar())
        trainer.run()
        train_iter['cnn'].reset()

    cnn.update_item_factors(mf.item_factor.W.data)
    _train_cnn()

    # train alternately
    for n in range(10):
        print('train alternately:', n)
        mf.update_convolution_item_factor(cnn, batch_size=cnn_batch_size)
        _train_mf()
        cnn.update_item_factors(mf.item_factor.W.data)
        _train_cnn()

    mf.to_cpu()
    cnn.to_cpu()
    serializers.save_npz('./result/convmf.npz', mf)
    serializers.save_npz('./result/cnn.npz', cnn)


def make_negative_test_case(ratings: List[RatingData], size: int) -> List[RatingData]:
    users = [r.user for r in ratings]
    items = [r.item for r in ratings]
    positive_cases = list(zip(users, items))
    sample_size = size + len(ratings)
    samplings = list(zip(np.random.choice(users, size=sample_size), np.random.choice(items, size=sample_size)))
    negative_cases = shuffle(list(set(samplings) - set(positive_cases)))[:size]
    return [RatingData(user=u, item=i, rating=0) for u, i in negative_cases]


def train_mf(batch_size: int, n_epoch: int, gpu: int, user_lambda: float, item_lambda: float):
    ratings = make_rating_data()
    n_factor = 300

    mf = ConvMF(ratings=ratings,
                n_factor=n_factor,
                user_lambda=user_lambda,
                item_lambda=item_lambda,
                descriptions=[],
                use_cnn=False)

    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()  # Make a specified GPU current
        mf.to_gpu()  # Copy the model to the GPU

    train_mf, test_mf = train_test_split(make_mf_data(ratings), test_size=0.1, random_state=123)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(mf)

    train_iter = iterators.SerialIterator(train_mf, batch_size, shuffle=True)
    test_iter = iterators.SerialIterator(test_mf, batch_size, repeat=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (n_epoch, 'epoch'), out='result')
    trainer.extend(extensions.Evaluator(test_iter, mf, device=gpu), name='test')
    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.PrintReport(entries=[
            'epoch',
            'main/loss',
            'test/main/loss',
            'elapsed_time']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()

    mf.to_cpu()
    serializers.save_npz('./result/mf.npz', mf)


# def train_mf():
#     ratings = make_rating_data()
#     n_factor = 300
#     n_trial = 10
#     user_lambda = 10
#     item_lambda = 100
#
#     n_item = len(pd.DataFrame(ratings).item.unique())
#     train, test = train_test_split(ratings, test_size=0.1, random_state=123)
#     model = MatrixFactorization(ratings=train, n_factor=n_factor, user_lambda=user_lambda, item_lambda=item_lambda, n_item=n_item)
#     for n in range(n_trial):
#         model.fit(n_trial=1)
#         predict = model.predict(users=[r.user for r in test], items=[r.item for r in test])
#         rmse = np.sqrt(np.mean(np.square(predict - np.array([r.rating for r in test]))))
#         print('rmse: %.4f' % rmse)
#
#     with open('./result/mf.pkl', 'wb') as f:
#         pickle.dump(model, f)

if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument('--mf-batch-size', type=int, default=1024)
    parser.add_argument('--cnn-batch-size', type=int, default=64)
    parser.add_argument('--n-epoch', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--n-out-channel', type=int, default=100)
    parser.add_argument('--user-lambda', type=float, default=10)
    parser.add_argument('--item-lambda', type=float, default=1)
    parser.add_argument('--n-factor', type=int, default=200)
    args = parser.parse_args()
    print(args)
    train_convmf(**vars(args))

# if __name__ == '__main__':
#     os.chdir(os.path.abspath(os.path.dirname(__file__)))
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--batch-size', type=int, default=1024)
#     parser.add_argument('--n-epoch', type=int, default=10)
#     parser.add_argument('--gpu', type=int, default=-1)
#     parser.add_argument('--user-lambda', type=float, default=10)
#     parser.add_argument('--item-lambda', type=float, default=100)
#     args = parser.parse_args()
#     print(args)
#     train_mf(**vars(args))
