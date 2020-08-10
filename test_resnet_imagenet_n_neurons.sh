#!/bin/bash

# add tf_models to PYTHONPATH
# Note: due to conflicts between tf_models and tf_models/gpu we will overwrite the PYTHONPATH
# and not append to it!

export PYTHONPATH=$PWD/tf_models/models
export PYTHONPATH=$PYTHONPATH:$PWD

sed -i 's/print_n_neurons = False/print_n_neurons = True/g' fs_coding.py

python3 tf_models/models/official/r1/resnet/imagenet_main.py \
	--eval_only \
	--data_dir=datasets/imagenet \
	--resnet_size=50 \
	--model_dir=checkpoints/imagenet/resnet50/checkpoint \
	--max_train_steps=1 \
	--use_fs | grep 'Number of neurons:' | tail -1


sed -i 's/print_n_neurons = True/print_n_neurons = False/g' fs_coding.py
