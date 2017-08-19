#  Copyright 2017 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script to train gradient navigation model
"""

import tensorflow as tf
from trainingData import GradientData
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns

BATCHSIZE = 50

if __name__ == "__main__":
    import mixedInputModel as mIM
    trainingData = GradientData.load("gd_training_data.hdf5")
    train_losses = []
    rank_errors = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(50000):
            batch_data = trainingData.training_batch(BATCHSIZE)
            xbatch = batch_data[0]
            ybatch = batch_data[1]
            cur_l = mIM.total_loss.eval(feed_dict={mIM.x_in: xbatch, mIM.y_: ybatch, mIM.keep_prob: 1.0})
            # compare ranks of options in prediction vs. ranks of real options
            pred = mIM.m_out.eval(feed_dict={mIM.x_in: xbatch, mIM.y_: ybatch, mIM.keep_prob: 1.0})
            sum_rank_diffs = 0.0
            for elem in range(BATCHSIZE):
                rank_real = np.unique(ybatch[elem, :], return_inverse=True)[1]
                rank_pred = np.unique(pred[elem, :], return_inverse=True)[1]
                sum_rank_diffs += np.sum(np.abs(rank_real - rank_pred))
            train_losses.append(cur_l)
            rank_errors.append(sum_rank_diffs / BATCHSIZE)
            if i % 200 == 0:
                print('step %d, training loss %g, rank loss %g' % (i, cur_l, sum_rank_diffs/BATCHSIZE))
            mIM.t_step.run(feed_dict={mIM.x_in: xbatch, mIM.y_: ybatch, mIM.keep_prob: mIM.DROP_TRAIN})
        weights_conv1 = mIM.W_conv1.eval()

    w_ext = np.max(np.abs(weights_conv1))
    fig, ax = pl.subplots(ncols=int(np.sqrt(mIM.N_CONV_LAYERS)), nrows=int(np.sqrt(mIM.N_CONV_LAYERS)), frameon=False,
                          figsize=(14, 2.8))
    ax = ax.ravel()
    for i, a in enumerate(ax):
        sns.heatmap(weights_conv1[:, :, 0, i], ax=a, vmin=-w_ext, vmax=w_ext, center=0, cbar=False)
        a.axis("off")
