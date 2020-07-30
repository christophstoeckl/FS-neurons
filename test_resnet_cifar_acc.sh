#!/bin/bash

# add tf_models to PYTHONPATH
# Note: due to conflicts between tf_models and tf_models/gpu we will overwrite the PYTHONPATH
# and not append to it!

export PYTHONPATH=$PWD/tf_models/models
export PYTHONPATH=$PYTHONPATH:$PWD

python3 tf_models/models/official/r1/resnet/cifar10_main.py \
	--data_dir=datasets/cifar \
	--eval_only \
	--resnet_size=50 \
	--model_dir=checkpoints/cifar/resnet50/checkpoint \
	--use_fs


