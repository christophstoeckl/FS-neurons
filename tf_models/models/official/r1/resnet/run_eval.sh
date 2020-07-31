#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "gpu number needed"
	exit 1
fi


CUDA_VISIBLE_DEVICES=$1 python3 imagenet_main.py \
--eval_only \
--data_dir=/calc/SHARED/imagenet/tf_records/ \
--resnet_size=50 \
--model_dir=/calc/stoeckl/binary_populations/checkpoints/resnet50/test2 \
--use_amos \
--max_train_steps=1000    # This also works for eval steps!!

