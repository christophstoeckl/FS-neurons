# FS-neurons

Official [FS-neurons](https://arxiv.org/pdf/2002.00860.pdf) github repository. 
Fs-neurons are compatible with TensorFlow 2.2. 


Take a look at the demo.py file to see how to use FS-neurons. 

To train an FS-neuron to approximate a new function please modify 
and use the find_coeffs.py file. 
Note: This file requires TensorFlow 1.14!


## Reproduce the results
One advantage of FS-coding how it can be easily integrated into existing models. 
The implementations of the models (ResNets, EfficientNet) are from the 
official [tensorflow models repository](https://github.com/tensorflow/models). 

### Hardware requirements


### Preparing the data

To reproduce the results it is necessary to download the cifar10 and the ImageNet2012 dataset. 

#### Cifar10
The cifar10 dataset can be downloaded by simply using the provided script with the command:

```bash
./download_cifar10
```

#### ImageNet
To download and prepare the ImageNet dataset, please follow the README located at:
`tf_models/models/research/slim/README.md`. 

#### EfficientNet pretrained parameters
The publicly available [pre-trained parameters](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) for the EfficientNet-B7 model can be 
downloaded using the command:
```bash
./download_effnet_weights.sh
```

### Reproduce test accuracy results
The accuary of the ResNet50 model consisting of FS-neurons can be reproduced using the command:
```bash
./test_resnet_cifar_acc.sh
```
Expected output:
Expected time:

The accuacy of the EfficientNet can be reproduced using the command:
```bash
./test_effnet_acc.sh
```
Expected output:
Expected time:


### Reproducing the average number of spikes
To compute the number of spikes, first open the file `fs_coding.py` and 
change the value of the variable `print_spikes` in line 10 from `False` to `True`. 
It is advisable to change this value back to `False` after completing the average spike count tests.


Then, run the following scripts to compute the average number of spikes per neuron. 


```bash
./test_resnet_cifar_spikes.sh
```
Expected output:
Expected time:

```bash
./test_effnet_spikes.sh
```

Expected output:
Expected time:

### Reproducing ResNet50 on ImageNet
To train 