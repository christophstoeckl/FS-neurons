import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation
import numpy as np
from fs_weights import *

original_relu = tf.nn.relu
original_sigmoid = tf.nn.sigmoid

print_spikes = False


# spike function
@tf.custom_gradient
def spike_function(v_scaled: tf.Tensor):
    z_ = tf.where(v_scaled > 0, tf.ones_like(v_scaled), tf.zeros_like(v_scaled))  # tf.nn.relu(tf.sign(v_scaled))
    z_ = tf.cast(z_, dtype=tf.float32)

    def grad(dy):
        dE_dz = dy
        dz_dv_scaled = tf.maximum(1 - tf.abs(v_scaled), 0)

        dE_dv_scaled = dE_dz * dz_dv_scaled

        return [dE_dv_scaled]

    return tf.identity(z_, name="SpikeFunction"), grad


# replacement functions:
def replace_sigmoid_with_fs():
    '''
    Call this function to replace the sigmoid functions in the tensorflow library by an 
    FS-neuron which approximates a sigmoid function. 
    '''
    original_keras_layer = layers.Activation

    def custom_layers(type_str):
        if type_str == "sigmoid":
            return fs_sigmoid
        else:
            return original_keras_layer(type_str)

    layers.Activation = custom_layers
    tf.nn.sigmoid = fs_sigmoid
    tf.sigmoid = fs_sigmoid


def replace_relu_with_fs():
    '''
    Call this function to replace the ReLU functions in the tensorflow library by an 
    FS-neuron which approximates a ReLU function. 
    '''
    original_keras_layer = layers.Activation

    def custom_layers(type_str):
        if type_str == "relu":
            return fs_relu
        else:
            return original_keras_layer(type_str)

    layers.Activation = custom_layers
    tf.nn.relu = fs_relu



def fs(x: tf.Tensor, h, d, T, K, return_reg=False):
    v = tf.identity(x)
    z = tf.zeros_like(x)
    out = tf.zeros_like(x)
    v_reg, z_reg, t = 0., 0., 0
    while_cond = lambda out, v_reg, z_reg, v, z, t: tf.less(t, K)

    def while_body(out, v_reg, z_reg, v, z, t):
        v_scaled = (v - T[t]) / (tf.abs(v) + 1)
        z = spike_function(v_scaled)  # spike function
        v_reg += tf.square(tf.reduce_mean(tf.maximum(tf.abs(v_scaled) - 1, 0)))  # regularization
        z_reg += tf.reduce_mean(z)
        if print_spikes:
            z = tf.Print(z, [tf.reduce_sum(z)])
        out += z * d[t]  # compute output
        v = v - z * h[t]  # update membrane potential
        t = t + 1
        return out, v_reg, z_reg, v, z, t

    ret = tf.while_loop(while_cond, while_body, [out, v_reg, z_reg, v, z, t])
    if return_reg:
        return ret[0:3]  # out, v_reg, z_reg
    else:
        return ret[0]  # out


def fs_swish(x: tf.Tensor, return_reg=False):
    with tf.name_scope("fs_Swish"):
        return fs(x, tf.constant(swish_h), tf.constant(swish_d), tf.constant(swish_T), K=len(swish_h),
                  return_reg=return_reg)


def fs_relu(x: tf.Tensor, n_neurons=6, v_max=25, return_reg=False, fast=False):
    '''
    Note: As the relu function is a special case, it is no necessary to use the fs() function. 
    It is computationally cheaper to simply discretize the input and clip to the 
    minimum and maximum.
    '''
    with tf.name_scope("fs_ReLU"):
        if fast:
            x = tf.maximum(x, 0)
            x /= v_max

            x *= 2 ** (n_neurons)
            i_out = tf.cast(tf.floor(x), tf.float32)
            i_out /= 2 ** (n_neurons)
            i_out *= v_max
            i_out = tf.minimum(i_out, v_max * (1 - 2 ** (-n_neurons)))
            if return_reg:
                return tf.identity(i_out, name="i_out"), tf.constant(1.)
            return tf.identity(i_out, name="i_out")
        else:
            return fs(x, tf.constant(relu_h), tf.constant(relu_d), tf.constant(relu_T),
                      K=len(relu_h), return_reg=return_reg)



def fs_sigmoid(x: tf.Tensor, return_reg=False):
    with tf.name_scope("fs_sigmoid"):
        return fs(x, tf.constant(sigmoid_h), tf.constant(sigmoid_d), tf.constant(sigmoid_T), K=len(sigmoid_h),
                  return_reg=return_reg)

