#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "gpu number needed"
	exit 1
fi


CUDA_VISIBLE_DEVICES=$1 python3 /calc/stoeckl/binary_populations/tf_models/tpu/models/official/efficientnet/main.py \
	--use_tpu=False  \
	--data_dir=/calc/SHARED/imagenet/tf_records  \
	--model_name=efficientnet-b7 \
	--eval_batch_size=1\
	--mode=eval  \
	--model_dir=/calc/stoeckl/binary_populations/checkpoints/efficientnet/official/efficientnet-b7-randaug \
	--num_eval_images=1 \
  --use_amos \
