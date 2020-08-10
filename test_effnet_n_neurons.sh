#!/bin/bash

# add tf_models to PYTHONPATH
# Note: due to conflicts between tf_models and tf_models/gpu we will overwrite the PYTHONPATH
# and not append to it!

export PYTHONPATH=$PWD/tf_models/tpu/models
export PYTHONPATH=$PYTHONPATH:$PWD

sed -i 's/print_n_neurons = False/print_n_neurons = True/g' fs_coding.py

python3 tf_models/tpu/models/official/efficientnet/main.py \
	--use_tpu=False \
	--data_dir=datasets/imagenet \
	--model_name=efficientnet-b7 \
	--eval_batch_size=1 \
	--mode=eval \
	--model_dir=checkpoints/imagenet/effnet-b7 \
	--num_eval_images=1 \
	--use_fs | grep 'Number of neurons:' | tail -1


sed -i 's/print_n_neurons = True/print_n_neurons = False/g' fs_coding.py
