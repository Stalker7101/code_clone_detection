#! /bin/bash

cd data
wget https://s3.amazonaws.com/code2vec/model/token_vecs.tar.gz
wget https://s3.amazonaws.com/code2vec/model/target_vecs.tar.gz

tar -xvf token_vecs.tar.gz
tar -xvf target_vecs.tar.gz

rm token_vecs.tar.gz target_vecs.tar.gz