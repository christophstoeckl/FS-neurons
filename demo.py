import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
import fs_coding as fs 

# replace tensorflow functions with fs_neurons function
fs.replace_relu_with_fs()
fs.replace_sigmoid_with_fs()

# do some computations with fs neurons
x = np.linspace(-5, 5, 1000, dtype=np.float32)

# Note, that the original tensorflow functions have been overwritten with FS-neuron functions. 
y_sigmoid = tf.nn.sigmoid(x)
y_relu = tf.nn.relu(x)

y_swish = fs.fs_swish(x)

# plot the results
plt.plot(x, y_sigmoid)
plt.plot(x, y_relu)
plt.plot(x, y_swish)
plt.show()

