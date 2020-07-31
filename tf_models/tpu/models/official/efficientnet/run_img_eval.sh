#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "gpu number needed"
	exit 1
fi

export MODEL=efficientnet-b7
CUDA_VISIBLE_DEVICES=$1 python3 eval_ckpt_main.py \
	--model_name=$MODEL \
	--ckpt_dir=/calc/stoeckl/binary_populations/checkpoints/efficientnet/official/efficientnet-b7-randaug/ \
	--example_img=/calc/SHARED/imagenet/validation/ILSVRC2012_val_00000006.JPEG \
	--labels_map_file=labels_map.txt \
	--use_amos
