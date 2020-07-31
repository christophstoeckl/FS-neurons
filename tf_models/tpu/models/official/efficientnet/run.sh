#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "gpu number needed"
	exit 1
fi


CUDA_VISIBLE_DEVICES=$1 python3 main.py \
	--use_tpu=False  \
	--data_dir=/calc/SHARED/imagenet/tf_records  \
	--model_name=efficientnet-b7 \
	--model_dir=/calc/stoeckl/binary_populations/checkpoints/efficientnet/test-b7 \
	--export_dir=/calc/stoeckl/binary_populations/checkpoints/efficientnet/tes-b7 \
	--train_batch_size=8 \
	--eval_batch_size=8 \
	--steps_per_eval=2000
