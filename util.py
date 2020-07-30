import tensorflow as tf
import numpy as np


@tf.custom_gradient
def spike_function(v_scaled):
    z_ = tf.where(v_scaled > 0, tf.ones_like(v_scaled), tf.zeros_like(v_scaled))
    z_ = tf.cast(z_, dtype=tf.float32)

    def grad(dy):
        dE_dz = dy
        dz_dv_scaled = tf.maximum(1 - tf.abs(v_scaled), 0)
        dE_dv_scaled = dE_dz * dz_dv_scaled
        return [dE_dv_scaled]

    return tf.identity(z_, name="SpikeFunction"), grad


def test_print(x):
    op = tf.print(x)
    with tf.control_dependencies([op]):
        return tf.identity(op)


def print_tensors(model):
    with tf.Session() as sess:
        import cv2
        im = np.float32(cv2.imread("automobile.png") / 255.)
        out = model(tf.constant(im[None, ...]))
        sess.run(tf.global_variables_initializer())
        sess.run(out)
        in_ten = tf.get_default_graph().get_tensor_by_name("input_1:0")
        coll = tf.get_collection("i_in")
        ret = sess.run(coll, feed_dict={in_ten: np.ones(shape=(1, 32, 32, 3))})
        for r in ret:
            print(np.max(r))


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def swish(x):
    return x * 1 / (1 + np.exp(-x))

