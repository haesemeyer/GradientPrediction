#  Copyright 2017 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Builds a gradient prediction network that uses a mixed input layer, i.e. the input is a 2D tensor of both behavior and
temperature variables on which convolution filters operate jointly
"""

import tensorflow as tf
import core


def create_dense_layer(name_sfx: str):
    """
    Creates a behavior specific dense layer
    :param name_sfx: Variable name suffix identifying behavior output
    :return: weigths, biases, layer units
    """
    w = core.create_weight_var("W_h_" + name_sfx, [N_CONV_LAYERS, N_DENSE], WDECAY)
    b = core.create_bias_var("B_h_" + name_sfx, [N_DENSE])
    h = tf.nn.relu(tf.matmul(h_conv1_flat, w) + b, "h_" + name_sfx)
    return w, b, h


def create_output(name_sfx: str, prev_out):
    """
    Creates behavior specific output layer
    :param name_sfx: Variable name suffix identifying behavior output
    :param prev_out: The output of the previous layer
    :return: weights, biases, output
    """
    w = core.create_weight_var("W_o_" + name_sfx, [N_DENSE, 1], WDECAY)
    b = core.create_bias_var("B_o_" + name_sfx, [1])
    out = tf.matmul(prev_out, w) + b
    return w, b, out


def create_behav_section(name_sfx: str):
    """
    Creates a behavior specific section of the model
    :param name_sfx: Variable name suffix identifying behavior output
    :return:
        [0]: Hidden layer: weights, biases, units
        [1]: Dropout layer
        [2]: Output layer: weights, biases, output
    """
    w_h, b_h, l_h = create_dense_layer(name_sfx)
    h_drop = tf.nn.dropout(l_h, keep_prob, name="h_drop_" + name_sfx)
    w_o, b_o, o = create_output(name_sfx, h_drop)
    return (w_h, b_h, l_h), h_drop, (w_o, b_o, o)


# Hyper parameters of the model
N_CONV_LAYERS = 40  # the number of convolution filters
N_DENSE = 512  # the number of units in the hidden layer
WDECAY = 1e-4  # weight decay constant
DROP_TRAIN = 0.5  # dropout probability during training

# globals
assert core.FRAME_RATE % core.MODEL_RATE == 0
t_bin = core.FRAME_RATE // core.MODEL_RATE  # bin input down to 5Hz
binned_size = core.FRAME_RATE * core.HIST_SECONDS // t_bin

# dropout probability placeholder
keep_prob = tf.placeholder(tf.float32)

# Shared network structure
# model input: BATCHSIZE x (Temp,Move,Turn) x HISTORYSIZE x 1 CHANNEL
x_in = tf.placeholder(tf.float32, [None, 3, core.FRAME_RATE*core.HIST_SECONDS, 1], "x_in")
# real outputs: BATCHSIZE x (dT(Stay), dT(Straight), dT(Left), dT(Right))
y_ = tf.placeholder(tf.float32, [None, 4], "y_")
# data binning layer
xin_pool = core.create_meanpool2d("xin_pool", x_in, 1, t_bin)
# convolution layer
W_conv1 = core.create_weight_var("W_conv1", [3, binned_size, 1, N_CONV_LAYERS])
B_conv1 = core.create_bias_var("B_conv1", [N_CONV_LAYERS])
conv1 = core.create_conv2d("conv1", xin_pool, W_conv1)
h_conv1 = tf.nn.relu(conv1 + B_conv1, "h_conv1")
h_conv1_flat = tf.reshape(h_conv1, [-1, N_CONV_LAYERS], "h_conv1_flat")

# Behavioral output specific network structure
# 1) Stay
hidden_stay, drop_stay, out_stay = create_behav_section("stay")
# 2) Straight
hidden_str, drop_str, out_str = create_behav_section("str")
# 3) Left
hidden_left, drop_left, out_left = create_behav_section("left")
# 4) Right
hidden_right, drop_right, out_right = create_behav_section("right")

# combine outputs
m_out = tf.concat([out_stay[2], out_str[2], out_left[2], out_right[2]], 1)

# get model loss and training step
total_loss = core.get_loss(y_, m_out)
t_step = core.create_train_step(y_, m_out)

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as pl
    import seaborn as sns
    print("Testing mixedInputModel", flush=True)
    print("For each 'behavior' subpart attempt to learn different sums on standard normal distribution", flush=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            xb1 = np.random.randn(100, 1, core.FRAME_RATE * core.HIST_SECONDS, 1)
            xb2 = xb1 ** 2
            xb3 = xb1 ** 3
            xbatch = np.concatenate((xb1, xb2, xb3), 1)
            ybatch = np.c_[np.sum(xb1 ** 2, axis=(1, 2)), np.sum((xb1/2) ** 2, axis=(1, 2)),
                           np.sum(xb1, axis=(1, 2)), np.sum(xb1 / 2, axis=(1, 2))]
            cur_l = total_loss.eval(feed_dict={x_in: xbatch, y_: ybatch, keep_prob: 1.0})
            cur_d = np.median(np.abs((ybatch - m_out.eval(feed_dict={x_in: xbatch, y_: ybatch, keep_prob: 1.0})) /
                                     ybatch))
            if i % 200 == 0:
                print('step %d, training loss %g, delta fraction %g' % (i, cur_l, cur_d))
            t_step.run(feed_dict={x_in: xbatch, y_: ybatch, keep_prob: DROP_TRAIN})
        weights_conv1 = W_conv1.eval()
        bias_conv1 = B_conv1.eval()

    w_ext = np.max(np.abs(weights_conv1))
    fig, ax = pl.subplots(ncols=int(np.sqrt(N_CONV_LAYERS)), nrows=int(np.sqrt(N_CONV_LAYERS)), frameon=False,
                          figsize=(14, 2.8))
    ax = ax.ravel()
    for i, a in enumerate(ax):
        sns.heatmap(weights_conv1[:, :, 0, i], ax=a, vmin=-w_ext, vmax=w_ext, center=0, cbar=False)
        a.axis("off")
