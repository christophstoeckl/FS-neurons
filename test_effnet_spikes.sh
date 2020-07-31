#!/bin/bash

# add tf_models to PYTHONPATH
# Note: due to conflicts between tf_models and tf_models/gpu we will overwrite the PYTHONPATH
# and not append to it!

export PYTHONPATH=$PWD/tf_models/tpu/models
export PYTHONPATH=$PYTHONPATH:$PWD

python3 tf_models/tpu/models/official/efficientnet/main.py \
	--use_tpu=False \
	--data_dir=datasets/imagenet \
	--model_name=efficientnet-b7 \
	--eval_batch_size=1 \
	--mode=eval \
	--model_dir=checkpoints/imagenet/effnet-b7 \
	--num_eval_images=50 \
	--use_fs 2> effnet_imagenet_spikes.txt

python3 extract_spikes.py \
	--file_name=effnet_imagenet_spikes.txt \
	--n_neurons=259000000 \
	--n_images=50


