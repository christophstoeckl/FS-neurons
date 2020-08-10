#!/bin/bash

# add tf_models to PYTHONPATH
# Note: due to conflicts between tf_models and tf_models/gpu we will overwrite the PYTHONPATH
# and not append to it!

export PYTHONPATH=$PWD/tf_models/models
export PYTHONPATH=$PYTHONPATH:$PWD

sed -i 's/print_spikes = False/print_spikes = True/g' fs_coding.py

# 32 train steps with batch size 32 each -> 1024 images
python3 tf_models/models/official/r1/resnet/imagenet_main.py \
	--eval_only \
	--data_dir=datasets/imagenet \
	--resnet_size=50 \
	--model_dir=checkpoints/imagenet/resnet50/checkpoint \
	--max_train_steps=32 \
	--use_fs \
	2> resnet_imagenet_spikes.txt

python extract_spikes.py \
	--file_name=resnet_imagenet_spikes.txt \
	--n_neurons=9600000 \
	--n_images=1024


sed -i 's/print_spikes = True/print_spikes = False/g' fs_coding.py
