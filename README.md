# FS-neurons

Official [FS-neurons](https://arxiv.org/pdf/2002.00860.pdf) github repository. 
Fs-neurons are compatible with TensorFlow 2.2. 


Take a look at the demo.py file to see how to use FS-neurons. 

To train an FS-neuron to approximate a new function please modify 
and use the `find_coeffs.py` file. 
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


### Verifying the number of neurons in the models.
In order to compute the average number of spikes, it is important to know the exact number of 
FS-neurons in the model. 

This number can be computed for the ResNet50 models using the command:
```bash
./test_resnet_cifar_n_neurons.sh
```
for obatining the number of FS-neurons used on the Cifar10 dataset and:
```bash
./test_resnet_imagenet_n_neurons.sh
```
for the number of FS-neurons used on the ImageNet dataset. 

The same can be done for the EfficientNet-B7 model:

```bash
./test_effnet_n_neurons.sh
```



### Reproducing the average number of spikes

Run the following scripts to compute the average number of spikes per neuron. 


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
Unfortunately the checkpoint file for the ResNet50 on ImageNet is too large for github.
Therefore it is necessary to train the model first. 

A new model can be trained using the command: 
```bash
./train_resnet_imagenet.sh
```


### Training the smaller ResNet versions on Cifar10

To train the smaller versions of the ResNet on the Cifar10 dataset one 
can simply modify and run the script located at:
`tf_models/models/official/r1/resnet/run_cifar10.sh`.

### Reproduce the Mean and Stddev of the average input to the FS-neuron

The result for the EfficientNet-B7 can be reproduced using the command: 
```bash
./test_effnet_mean_stddev.sh
```

In a similar fashion, these results for the ResNet50 can be reproduced using the command: 
```bash 
./test_resnet_imagenet_mean_stddev.sh
```


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


