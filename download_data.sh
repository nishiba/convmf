#!/usr/bin/env bash

mkdir data
wget http://files.grouplens.org/datasets/movielens/ml-10m.zip -O ./data/ml-10m.zip
unzip ./data/ml-10m.zip -d ./data/
