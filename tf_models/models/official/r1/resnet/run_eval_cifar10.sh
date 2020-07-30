#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "gpu number needed"
	exit 1
fi


CUDA_VISIBLE_DEVICES=$1 python3 cifar10_main.py \
--data_dir=/calc/SHARED/CIFAR10 \
--eval_only \
--resnet_size=50  \
--model_dir=/calc/stoeckl/binary_populations/checkpoints/cifar10/resnet50/test3 \
--use_amos

