#!/bin/bash

python3 tf_models/models/official/r1/resnet/cifar10_download_and_extract.py --data_dir=datasets/cifar
mv datasets/cifar/cifar-10-batches-bin/* datasets/cifar
