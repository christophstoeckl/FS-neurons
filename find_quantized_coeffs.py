import tensorflow as tf
import os
import sys
import numpy as np
from util import *
import datetime
from absl import flags
from absl import app
FLAGS = flags.FLAGS


def main(_):
    K = FLAGS.K
    Q = FLAGS.Q
    batch_size = 10000
    v_min, v_max = -8, 12  # sigmoid -10, 10  #  -8, 12  # swish
    name = 'sigmoid'
    quant_min = -8
    quant_max = 8

    x = np.linspace(v_min, v_max, batch_size)

    y = sigmoid(x)

    def quantize(x, q):
        return tf.quantization.fake_quant_with_min_max_args(x, min=quant_min, max=quant_max, num_bits=q)

    with tf.Session() as sess:
        x_in = tf.placeholder(shape=batch_size, dtype=tf.float32)
        y_in = tf.placeholder(shape=y.shape, dtype=tf.float32)
        K_in = tf.placeholder(shape=(), dtype=tf.int32)
        is_training = tf.placeholder(shape=(), dtype=tf.bool)

        h = tf.Variable(tf.random.uniform(shape=(K,), minval=-1, maxval=0, dtype=tf.float32))
        d = tf.Variable(tf.random.uniform(shape=(K,), minval=-0.5, maxval=1, dtype=tf.float32))
        T = tf.Variable(tf.random.uniform(shape=(K,), minval=-1, maxval=1, dtype=tf.float32))

        h = tf.cond(is_training, lambda: h, lambda: quantize(h, Q))
        d = tf.cond(is_training, lambda: d, lambda: quantize(d, Q))
        T = tf.cond(is_training, lambda: T, lambda: quantize(T, Q))

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
                z_reg += tf.reduce_mean(z)
                out += z * d[t]  # compute output
                t = t + 1
                return out, v_reg, z_reg, v, z, t

            ret = tf.while_loop(while_cond, while_body, [out, v_reg, z_reg, v, z, t])
            return ret[0:3]  # out, v_reg, z_reg

        y_approx, v_reg, z_reg = fs_coding(x_in, h, d, T, K_in)

        loss = tf.reduce_mean(tf.pow(tf.abs(y_in - y_approx), 2.)) + \
               tf.random.uniform(shape=(), minval=0.08, maxval=0.12, dtype=tf.float32) * v_reg + 0.00 * z_reg

        lr = tf.random.uniform(shape=(), minval=0.09, maxval=0.5, dtype=tf.float32)
        train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)  # 0.01
        best = 1000000
        for gen in range(1000):
            sess.run(tf.global_variables_initializer())
            i = 0
            current_best = 100000
            while i < 300:
                i += 1
                sess.run([train_step], feed_dict={x_in: x, y_in: y, K_in: K, is_training: True})
                # eval
                l, vrl, zrl, y_res = sess.run([loss, v_reg, z_reg, y_approx],
                                              feed_dict={x_in: x, y_in: y, K_in: K, is_training: False})
                if l < current_best:
                    current_best = l
                    i = i if i < 0 else i - 50

                if l < best:
                    h_np, d_np, T_np = sess.run([h, d, T], feed_dict={is_training: False})
                    best = l
                    sys.stdout.flush()

            print(f"K: {K}, quant: {Q} Gen: {gen} Time: {datetime.datetime.now()} Loss: {l} (v:{vrl},z:{zrl} )")
            print(
                f"{name}_h = {np.array2string(h_np, separator=',')}\n{name}_d = {np.array2string(d_np, separator=',')}\n"
                f"{name}_T = {np.array2string(T_np, separator=',')}\n\n")
            sys.stdout.flush()


if __name__ == '__main__':
    flags.DEFINE_integer('K', 4, 'K')
    flags.DEFINE_integer('Q', 4, 'Q')
    app.run(main)
    pass

