#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "gpu number needed"
	exit 1
fi


CUDA_VISIBLE_DEVICES=$1 python3 cifar10_main.py \
--data_dir=/calc/SHARED/CIFAR10 \
--resnet_size=14 \
--model_dir=/calc/stoeckl/binary_populations/checkpoints/cifar10/resnet14/test1

