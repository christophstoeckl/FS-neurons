#!/bin/bash

python3 tf_models/models/official/r1/resnet/imagnet_main.py \
	--data_dir=datasets/imagenet \
	--resnet_size=50 \
	--model_dir=checkpoints/imagenet/resnet50 \
	--resnet_version=1
