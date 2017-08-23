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
from scipy.ndimage import gaussian_filter1d
import h5py

BATCHSIZE = 50
TESTSIZE = 200

EVAL_TRAIN_EVERY = 5  # every this many trials training set performance is evaluated
EVAL_TEST_EVERY = 1000  # every this many trials test set performance is evaluated

if __name__ == "__main__":
    import mixedInputModel as mIM
    trainingData = GradientData.load("gd_training_data.hdf5")
    testData = GradientData.load("gd_test_data_1.hdf5")
    # enforce same scaling on testData as on trainingData
    testData.ang_mean = trainingData.ang_mean
    testData.ang_std = trainingData.ang_std
    testData.disp_mean = trainingData.disp_mean
    testData.disp_std = trainingData.disp_std
    testData.temp_mean = trainingData.temp_mean
    testData.temp_std = trainingData.temp_std
    train_losses = []
    rank_errors = []
    test_losses = []
    test_rank_errors = []
    # store our training operation
    tf.add_to_collection('train_op', mIM.t_step)
    # create saver
    saver = tf.train.Saver(max_to_keep=None)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100001):
            # save naive model including full graph
            if i == 0:
                save_path = saver.save(sess, "./model_data/mixedInputModel.ckpt", global_step=i)
                print("Model saved in file: %s" % save_path)
            # save variables every 5000 steps but don't re-save model-meta
            if i != 0 and i % 5000 == 0:
                save_path = saver.save(sess, "./model_data/mixedInputModel.ckpt", global_step=i, write_meta_graph=False)
                print("Model saved in file: %s" % save_path)
            batch_data = trainingData.training_batch(BATCHSIZE)
            xbatch = batch_data[0]
            ybatch = batch_data[1]
            # every five steps compute training losses
            if i % EVAL_TRAIN_EVERY == 0:
                cur_l = mIM.sq_loss.eval(feed_dict={mIM.x_in: xbatch, mIM.y_: ybatch, mIM.keep_prob: 1.0})
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
            # every 1000 steps run test
            if i % EVAL_TEST_EVERY == 0:
                test = testData.training_batch(TESTSIZE)
                xtest = test[0]
                ytest = test[1]
                cur_l = mIM.sq_loss.eval(feed_dict={mIM.x_in: xtest, mIM.y_: ytest, mIM.keep_prob: 1.0})
                pred_test = mIM.m_out.eval(feed_dict={mIM.x_in: xtest, mIM.y_: ytest, mIM.keep_prob: 1.0})
                sum_rank_diffs = 0.0
                for elem in range(TESTSIZE):
                    rank_real = np.unique(ytest[elem, :], return_inverse=True)[1]
                    rank_pred = np.unique(pred_test[elem, :], return_inverse=True)[1]
                    sum_rank_diffs += np.sum(np.abs(rank_real - rank_pred))
                print("TEST")
                print('step %d, test loss %g, rank loss %g' % (i, cur_l, sum_rank_diffs / TESTSIZE))
                print("TEST")
                test_losses.append(cur_l)
                test_rank_errors.append(sum_rank_diffs / TESTSIZE)

            mIM.t_step.run(feed_dict={mIM.x_in: xbatch, mIM.y_: ybatch, mIM.keep_prob: mIM.KEEP_TRAIN})
        weights_conv1 = mIM.W_conv1.eval()

    w_ext = np.max(np.abs(weights_conv1))
    fig, ax = pl.subplots(ncols=int(np.sqrt(mIM.N_CONV_LAYERS)), nrows=int(np.sqrt(mIM.N_CONV_LAYERS)), frameon=False,
                          figsize=(14, 2.8))
    ax = ax.ravel()
    for i, a in enumerate(ax):
        sns.heatmap(weights_conv1[:, :, 0, i], ax=a, vmin=-w_ext, vmax=w_ext, center=0, cbar=False)
        a.axis("off")
    fig.savefig("ConvolutionFilters.pdf", type="pdf")

    fig = pl.figure()
    pl.plot(np.arange(len(train_losses)) * EVAL_TRAIN_EVERY, train_losses, 'o', alpha=0.2)
    pl.plot(np.arange(len(train_losses)) * EVAL_TRAIN_EVERY, gaussian_filter1d(train_losses, 25))
    pl.plot(np.arange(len(test_losses)) * EVAL_TEST_EVERY, test_losses, 'o')
    pl.xlabel("Batch")
    pl.ylabel("Training/Test loss")
    sns.despine()
    fig.savefig("SquaredLoss.pdf", type="pdf")

    fig = pl.figure()
    pl.plot(np.arange(len(rank_errors)) * EVAL_TRAIN_EVERY, rank_errors, 'o', alpha=0.2)
    pl.plot(np.arange(len(rank_errors)) * EVAL_TRAIN_EVERY, gaussian_filter1d(rank_errors, 25))
    pl.plot(np.arange(len(test_rank_errors)) * EVAL_TEST_EVERY, test_rank_errors, 'o')
    pl.xlabel("Batch")
    pl.ylabel("Avg. rank error")
    sns.despine()
    fig.savefig("RankError.pdf", type="pdf")

    # save loss evaluations to file
    with h5py.File("./model_data/losses.hdf5", "x") as dfile:
        dfile.create_dataset("train_eval", data=np.arange(len(train_losses)) * EVAL_TRAIN_EVERY)
        dfile.create_dataset("train_losses", data=train_losses)
        dfile.create_dataset("test_eval", data=len(test_losses) * EVAL_TEST_EVERY)
        dfile.create_dataset("test_losses", data=test_losses)
        dfile.create_dataset("train_rank_errors", data=rank_errors)
        dfile.create_dataset("test_rank_errors", data=test_rank_errors)
