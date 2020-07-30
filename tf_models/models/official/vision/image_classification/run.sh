#!/bin/bash


CUDA_VISIBLE_DEVICES=3 python3 resnet_imagenet_main.py \
 --data_dir=/calc/SHARED/tensorflow_datasets/imagenet2012/2.0.1 \
 --model_dir=/calc/stoeckl/binary_populations/checkpoints/imagenet
