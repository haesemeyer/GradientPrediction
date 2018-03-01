#  Copyright 2017 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script to train gradient navigation model
"""

from core import GradientData, ZfGpNetworkModel
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import h5py


BATCHSIZE = 32  # the sample size of each training batch
TESTSIZE = 128  # the sample size of each test batch
N_EPOCHS = 10  # the number of training epochs to run

EVAL_TRAIN_EVERY = 5  # every this many trials training set performance is evaluated
EVAL_TEST_EVERY = 1000  # every this many trials test set performance is evaluated

SEPARATE = True

if SEPARATE:
    N_UNITS = [512, 512]
    N_BRANCH = 2
    N_MIXED = 3
    N_CONV = 40
    chk_file = "./model_data/separateInputModel.ckpt"
else:
    N_UNITS = 512
    N_BRANCH = 0
    N_MIXED = 3
    N_CONV = 40
    chk_file = "./model_data/mixedInputModel.ckpt"


def train_one(batch, net_model: ZfGpNetworkModel):
    # save variables every 10000 steps but don't re-save model-meta
    if global_count != 0 and global_count % 50000 == 0:
        path = net_model.save_state(chk_file, global_count, False)
        print("Model saved in file: %s" % path)
    xbatch = batch[0]
    ybatch = batch[1]
    # every five steps compute training losses
    if global_count % EVAL_TRAIN_EVERY == 0:
        cur_l = net_model.get_squared_loss(xbatch, ybatch)
        # compare ranks of options in prediction vs. ranks of real options
        pred = net_model.predict(xbatch)
        sum_rank_diffs = 0.0
        for elem in range(BATCHSIZE):
            rank_real = np.unique(ybatch[elem, :], return_inverse=True)[1]
            rank_pred = np.unique(pred[elem, :], return_inverse=True)[1]
            sum_rank_diffs += np.sum(np.abs(rank_real - rank_pred))
        train_losses.append(cur_l)
        rank_errors.append(sum_rank_diffs / BATCHSIZE)
        if global_count % 200 == 0:
            print('step %d of %d, training loss %g, rank loss %g' % (global_count, total_steps, cur_l,
                                                                     sum_rank_diffs / BATCHSIZE))
    # run test
    if global_count % EVAL_TEST_EVERY == 0:
        test = testData.training_batch(TESTSIZE)
        xtest = test[0]
        ytest = test[1]
        cur_l = net_model.get_squared_loss(xtest, ytest)
        pred_test = net_model.predict(xtest)
        sum_rank_diffs = 0.0
        for elem in range(TESTSIZE):
            rank_real = np.unique(ytest[elem, :], return_inverse=True)[1]
            rank_pred = np.unique(pred_test[elem, :], return_inverse=True)[1]
            sum_rank_diffs += np.sum(np.abs(rank_real - rank_pred))
        print("TEST")
        print('step %d, test loss %g, rank loss %g' % (global_count, cur_l, sum_rank_diffs / TESTSIZE))
        print("TEST")
        test_losses.append(cur_l)
        test_rank_errors.append(sum_rank_diffs / TESTSIZE)
    net_model.train(xbatch, ybatch)


if __name__ == "__main__":
    trainingData_1 = GradientData.load("gd_training_data.hdf5")
    trainingData_2 = GradientData.load("gd_training_data_rev.hdf5")
    trainingData_2.copy_normalization(trainingData_1)
    testData = GradientData.load("gd_test_data_radial.hdf5")
    # enforce same scaling on testData as on trainingData
    testData.copy_normalization(trainingData_1)
    epoch_1_size = trainingData_1.data_size // BATCHSIZE
    epoch_2_size = trainingData_2.data_size // BATCHSIZE
    train_list = []  # this list will contain 2 data/epoch_size tuples to allow training of both sets in random order

    train_losses = []
    rank_errors = []
    test_losses = []
    test_rank_errors = []
    global_count = 0
    total_steps = N_EPOCHS * (epoch_1_size + epoch_2_size)
    with ZfGpNetworkModel() as Model:
        Model.setup(N_CONV, N_UNITS, N_BRANCH, N_MIXED)
        # save naive model including full graph
        save_path = Model.save_state(chk_file, 0)
        print("Model saved in file: %s" % save_path)
        for epoch in range(N_EPOCHS):
            # determine this epoch's training order
            if np.random.rand() < 0.5:
                train_list = [(trainingData_1, epoch_1_size), (trainingData_2, epoch_2_size)]
            else:
                train_list = [(trainingData_2, epoch_2_size), (trainingData_1, epoch_1_size)]
            for tstep1 in range(train_list[0][1]):
                # train on first data
                batch_data = train_list[0][0].training_batch(BATCHSIZE)
                train_one(batch_data, Model)
                # update our global step counter
                global_count += 1
            for tstep2 in range(train_list[1][1]):
                # train in second data
                batch_data = train_list[1][0].training_batch(BATCHSIZE)
                train_one(batch_data, Model)
                # update our global step counter
                global_count += 1
        # save final progress
        save_path = Model.save_state(chk_file, global_count, False)
        print("Final model saved in file: %s" % save_path)
        weights_conv1 = Model.convolution_data[0]
        if 't' in weights_conv1:
            weights_conv1 = weights_conv1['t']
        else:
            weights_conv1 = weights_conv1['m']

    w_ext = np.max(np.abs(weights_conv1))
    fig, ax = pl.subplots(ncols=int(np.sqrt(N_CONV)), nrows=int(np.sqrt(N_CONV)), frameon=False,
                          figsize=(14, 2.8))
    ax = ax.ravel()
    for j, a in enumerate(ax):
        sns.heatmap(weights_conv1[:, :, 0, j], ax=a, vmin=-w_ext, vmax=w_ext, center=0, cbar=False)
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
        dfile.create_dataset("test_eval", data=np.arange(len(test_losses)) * EVAL_TEST_EVERY)
        dfile.create_dataset("test_losses", data=test_losses)
        dfile.create_dataset("train_rank_errors", data=rank_errors)
        dfile.create_dataset("test_rank_errors", data=test_rank_errors)
