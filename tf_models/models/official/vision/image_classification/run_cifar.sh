#!/bin/bash


CUDA_VISIBLE_DEVICES=3 python3 resnet_cifar_main.py --data_dir=/calc/SHARED/CIFAR10 --model_dir=/calc/stoeckl/binary_populations/checkpoints/test 
