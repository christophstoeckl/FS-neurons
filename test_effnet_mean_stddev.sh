#!/bin/bash

# add tf_models to PYTHONPATH
# Note: due to conflicts between tf_models and tf_models/gpu we will overwrite the PYTHONPATH
# and not append to it!

export PYTHONPATH=$PWD/tf_models/tpu/models
export PYTHONPATH=$PYTHONPATH:$PWD

sed -i 's/print_mean_stddev = False/print_mean_stddev = True/g' fs_coding.py

python3 tf_models/tpu/models/official/efficientnet/main.py \
	--use_tpu=False \
	--data_dir=datasets/imagenet \
	--model_name=efficientnet-b7 \
	--eval_batch_size=1 \
	--mode=eval \
	--model_dir=checkpoints/imagenet/effnet-b7 \
	--num_eval_images=100 \
	--use_fs 2> effnet_imagenet_mean_stddev.txt

python3 extract_mean_stddev.py \
	--file_name=effnet_imagenet_mean_stddev.txt \

sed -i 's/print_mean_stddev = True/print_mean_stddev = False/g' fs_coding.py
