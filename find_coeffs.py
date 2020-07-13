import tensorflow as tf
import os
import sys
import numpy as np
from util import *
import datetime

# hyperparameters
K = 10
batch_size = 100000
n_gen = 100

# minimum and maximum which should be approximated.
x_min, x_max = -10, 10
name = 'sigmoid'

def inter(x):
    ret = np.interp(x,
                    xp=[x_min, mean - 3 * stddev, mean - 2 * stddev, mean, mean + 2 * stddev, mean + 3 * stddev, x_max],
                    fp=[0.5, 1, 10, 11, 10, 1, 0.5])
    return ret



x = np.linspace(x_min, x_max, batch_size)
# change this to the function you want to approximate
y = sigmoid(x)

mean = 0
stddev = 2

# the importance allows the user to add weights to the loss function.
# this way, certain regions can be made more important.
imp = 1


def fs_coding(x, h, d, T, K):
    v = tf.identity(x)
    z = tf.zeros_like(x)
    out = tf.zeros_like(x)
    v_reg, z_reg, t = 0., 0., 0
    while_cond = lambda out, v_reg, z_reg, v, z, t: tf.less(t, K)

    def while_body(out, v_reg, z_reg, v, z, t):
        v = v - z * h[t]  # update membrane potential
        v_scaled = (v - T[t]) / (tf.abs(v) + 1)
        z = spike_function(v_scaled)  # spike function
        v_reg += tf.square(tf.reduce_mean(tf.maximum(tf.abs(v_scaled) - 1, 0)))  # regularization
        z_reg += tf.reduce_mean(imp * z)
        out += z * d[t]  # compute output
        t = t + 1
        return out, v_reg, z_reg, v, z, t

    ret = tf.while_loop(while_cond, while_body, [out, v_reg, z_reg, v, z, t])
    return ret[0:3]  # out, v_reg, z_reg


x_in = tf.placeholder(shape=batch_size, dtype=tf.float32)
y_in = tf.placeholder(shape=y.shape, dtype=tf.float32)
K_in = tf.placeholder(shape=(), dtype=tf.int32)

h = tf.Variable(tf.random.uniform(shape=(K,), minval=-1, maxval=0, dtype=tf.float32))
d = tf.Variable(tf.random.uniform(shape=(K,), minval=-0.5, maxval=1, dtype=tf.float32))
T = tf.Variable(tf.random.uniform(shape=(K,), minval=-1, maxval=1, dtype=tf.float32))

y_approx, v_reg, z_reg = fs_coding(x_in, h, d, T, K_in)

# Loss is a mean squared error with additional voltage and spike regularization terms.
loss = tf.reduce_mean(imp * tf.pow(tf.abs(y_in - y_approx), 2.)) + \
       tf.random.uniform(shape=(), minval=0.08, maxval=0.12, dtype=tf.float32) * v_reg + 0.00 * z_reg

lr = tf.random.uniform(shape=(), minval=0.09, maxval=0.5, dtype=tf.float32)
train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

best = 100000
loss_hist = []

with tf.Session() as sess:
    for gen in range(n_gen):
        sess.run(tf.global_variables_initializer())
        i = 0
        current_best = 100000
        while i < 5000:
            i += 1
            _, l, vrl, zrl, y_res = sess.run([train_step, loss, v_reg, z_reg, y_approx], feed_dict={x_in: x, y_in: y, K_in: K})
            loss_hist.append(l)
            if l < current_best:
                current_best = l
                i = i if i < 0 else i - 1000

            if l < best:
                print(f"K: {K} Gen: {gen}Time: {datetime.datetime.now()} Loss: {l} (v:{vrl},z:{zrl} )")
                h_np, d_np, T_np = sess.run([h, d, T])
                print(
                    f"{name}_h = {np.array2string(h_np, separator=',')}\n{name}_d = {np.array2string(d_np, separator=',')}\n"
                    f"{name}_T = {np.array2string(T_np, separator=',')}\n\n")
                best = l
                sys.stdout.flush()
