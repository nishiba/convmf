#!/usr/bin/env bash

mkdir data
wget http://files.grouplens.org/datasets/movielens/ml-10m.zip -O ./data/ml-10m.zip
tar -zxvf ./data/ml-10m.zip -C ./data/
