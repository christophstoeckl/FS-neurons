# FS-neurons

Official [FS-neurons](https://arxiv.org/pdf/2002.00860.pdf) github repository. 
Fs-neurons are compatible with TensorFlow 2.2. 


Take a look at the demo.py file to see how to use FS-neurons. 

To train an FS-neuron to approximate a new function please modify 
and use the `find_coeffs.py` file. 
Note: This codebase requires TensorFlow 1.14!


## Reproduce the results
One advantage of FS-coding how it can be easily integrated into existing models. 
The implementations of the models (ResNets, EfficientNet) are from the 
official [tensorflow models repository](https://github.com/tensorflow/models). 

### Hardware requirements
This software has been tested on a Quadro RTX 6000 GPU. 


### Software requirements
This software has been tested under Debian GNU/Linux bullseye/sid. 

A list of software requirements can be found in the `requirements.txt` file.

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

### Reproducing ResNet50 on ImageNet
Unfortunately the checkpoint file for the ResNet50 on ImageNet is too large for github.
Therefore it is necessary to train the model first, if you are working with a cloned version 
of this repository. 

A new model can be trained using the command: 
```bash
./train_resnet_imagenet.sh
```


### Reproduce test accuracy results
The accuary of the ResNet50 model consisting of FS-neurons can be reproduced using the command:
```bash
./test_resnet_cifar_acc.sh
```
Expected output: `INFO:tensorflow:...  accuracy = 0.925, accuracy_top_5 = 0.9959, global_step = 114085, loss = 0.5251896 

Expected runtime:` 42 seconds

To test the ResNet50 model on ImageNet use the following command:
```bash
./test_resnet_imagenet_acc.sh
```
Expected output: `INFO:tensorflow:...  accuracy = 0.7546875, accuracy_top_5 = 0.921875, global_step = 2642442, loss = 1.5057617`

Expected runtime: 1 min, 48 sec


The accuacy of the EfficientNet can be reproduced using the command:
```bash
./test_effnet_acc.sh
```
Expected output: `INFO:tensorflow:... loss = 1.9050122, top_1_accuracy = 0.844`

Only 500 test images are evaluated, therefore the performance might vary slightly. 
This number can be increased by modifying this script.

Expected runtime: 8 min, 40 sec


### Verifying the number of neurons in the models.
In order to compute the average number of spikes, it is important to know the exact number of 
FS-neurons in the model. 

This number can be computed for the ResNet50 model on Cifar10 using the command:
```bash
./test_resnet_cifar_n_neurons.sh
```
Expected output: `Number of neurons: 475136`

Expected runtime: 42 sec

for obatining the number of FS-neurons used on the ImageNet dataset:
```bash
./test_resnet_imagenet_n_neurons.sh
```
Expected output: `Number of neurons: 9608704`

Expected runtime: 33 sec

for the number of FS-neurons used on the ImageNet dataset. 

The same can be done for the EfficientNet-B7 model:

```bash
./test_effnet_n_neurons.sh
```
Expected output: `Number of neurons: 259366992`

Expected runtime: 1 min 37 sec

### Reproducing the average number of spikes

Run the following scripts to compute the average number of spikes per neuron. 


```bash
./test_resnet_cifar_spikes.sh
```
Expected output: `Average number of spikes: 1.3624396859004582`

Expected runtime: 1 min 40 sec

```bash
./test_effnet_spikes.sh
```

Expected output: `Average number of spikes: 2.142410090965251`

Expected runtime: 6 min 22 sec

### Training the smaller ResNet versions on Cifar10

To train the smaller versions of the ResNet on the Cifar10 dataset one 
can simply modify and run the script located at:
`tf_models/models/official/r1/resnet/run_cifar10.sh`.

### Reproduce the Mean and Stddev of the average input to the FS-neuron

The result for the EfficientNet-B7 can be reproduced using the command: 
```bash
./test_effnet_mean_stddev.sh
```
Expected output:
`Average mean of Input: -0.0631789654635205
Average stddev of Input: 1.563287000375583`


Expected runtime: 6 min 36 sec


In a similar fashion, these results for the ResNet50 can be reproduced using the command: 
```bash 
./test_resnet_imagenet_mean_stddev.sh
```
Expected output:
`Average mean of Input: -0.3588376833517857
Average stddev of Input: 1.4623265127538263`


Expected runtime: 2 min 49 sec

## Finding FS-neuron parameters

The parameters for the Swish and the sigmoid function can be found in the 
`fs_weights.py` file, along with the ideal parameters for the ReLU function. 

To approximate another activation function using FS-neurons it is necessary to find 
the new internal parameters first. 

The file `find_coeffs.py` can be used to find a new set of parameters. 
By changing the value of the variable `y` in the script, new activation functions can be approximated. 
It is also possible to add more importance to a certain region of the function, using the `imp` variable. 

In some scenarios it might be advantageous to have quantized parameters, possibly due to hardware constraints. 

In this case, the file `find_quantized_coeffs.py` can be used instead. 


