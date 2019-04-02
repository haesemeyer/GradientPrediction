#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script for panels of single-neuron response predictivity by ANN units - for Figure S2
"""


import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pl
import seaborn as sns
from Figure3 import mpath
from mo_types import MoTypes
import core as c
import analysis as a
from sklearn.linear_model import Ridge
import h5py
from global_defs import GlobalDefs
from scipy.signal import convolve


# file definitions
base_path = "./model_data/Adam_1e-4/sepInput_mixTrain/"
paths_512 = [f + '/' for f in os.listdir(base_path) if "_3m512_" in f]


if __name__ == "__main__":
    save_folder = "./DataFigures/NeuronPredictivityPanels/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42

    # load zebrafish activity data
    dfile = h5py.File('zebrafish_brain_data.hdf5', 'r')
    all_activity = np.array(dfile['all_activity'])
    no_nan_aa = np.array(dfile['no_nan_aa'])
    tf_centroids = np.array(dfile['tf_centroids'])[no_nan_aa, :]
    zf_clusters = np.array(dfile['membership'])[no_nan_aa]
    dfile.close()

    std = c.GradientData.load_standards("gd_training_data.hdf5")
    mo = MoTypes(False)
    ana = a.Analyzer(mo, std, "sim_store.hdf5", "activity_store.hdf5")
    # load activity clusters from file
    clfile = h5py.File("cluster_info.hdf5", "r")
    clust_ids = np.array(clfile["clust_ids"])
    clfile.close()

    # load and interpolate temperature stimulus
    dfile = h5py.File("stimFile.hdf5", 'r')
    tsin = np.array(dfile['sine_L_H_temp'])
    x = np.arange(tsin.size)  # stored at 20 Hz !
    xinterp = np.linspace(0, tsin.size, tsin.size * GlobalDefs.frame_rate // 20)
    temperature = np.interp(xinterp, x, tsin)
    dfile.close()

    # load stack types
    dfile = h5py.File("stack_types.hdf5", 'r')
    stack_types = np.array(dfile["stack_types"])[no_nan_aa]
    dfile.close()

    # get activity data
    all_cells = []
    all_ids = []
    for i, p in enumerate(paths_512):
        cell_res, ids = ana.temperature_activity(mpath(base_path, p), temperature, i)
        all_cells.append(cell_res)
        all_ids.append(ids)
    all_cells = np.hstack(all_cells)
    all_ids = np.hstack(all_ids)

    # convolve activity with nuclear gcamp calcium kernel
    tau_on = 1.4  # seconds
    tau_on *= GlobalDefs.frame_rate  # in frames
    tau_off = 2  # seconds
    tau_off *= GlobalDefs.frame_rate  # in frames
    kframes = np.arange(10 * GlobalDefs.frame_rate)  # 10 s long kernel
    kernel = 2 ** (-kframes / tau_off) * (1 - 2 ** (-kframes / tau_on))
    kernel = kernel / kernel.sum()
    # convolve with our kernel
    for i in range(all_cells.shape[1]):
        all_cells[:, i] = convolve(all_cells[:, i], kernel, mode='full')[:all_cells.shape[0]]

    # normalize fish and network responses
    all_activity = all_activity.T
    all_activity -= np.mean(all_activity, 0, keepdims=True)
    all_activity /= np.std(all_activity, 0, keepdims=True)
    all_cells = all_cells[::20, :]  # reduce ANN cells to fish timebase of 5Hz by subsampling
    all_cells -= np.mean(all_cells, 0, keepdims=True)
    all_cells /= np.std(all_cells, 0, keepdims=True)

    # obtain fit and test data
    # limit predictors to one network model
    net_model = 0
    units_to_use = all_ids[0, :] == net_model
    # optionally limit the fraction of fish cells to perform prediction on
    cells_to_fit = np.random.rand(all_activity.shape[1]) < 1  # means use all cells
    cell_centroids = tf_centroids[cells_to_fit, :]
    cell_types = stack_types[cells_to_fit]  # to not plot trigeminal cells onto our brain map
    # divide data by trials to get fit and test data portions for network and fish
    fish_to_fit = all_activity[:all_activity.shape[0]//3*2, cells_to_fit]
    fish_to_test = all_activity[all_activity.shape[0]//3*2:, cells_to_fit]
    netw_to_fit = all_cells[:fish_to_fit.shape[0], units_to_use]
    valid = (np.sum(np.isnan(netw_to_fit), 0) == 0)  # exlude cells with NaN's in traces (those which had 0 activity)
    netw_to_fit = netw_to_fit[:, valid]
    netw_to_test = all_cells[fish_to_fit.shape[0]:, units_to_use]
    netw_to_test = netw_to_test[:, valid]

    # create ridge regression object and obtain per-cell scores
    rdg = Ridge(alpha=0.01)
    rdg.fit(netw_to_fit, fish_to_fit)
    predictions = rdg.predict(netw_to_test)
    scores = np.array([np.corrcoef(p, f)[0, 1]**2 for (p, f) in zip(predictions.T, fish_to_test.T)])

    # compute self-prediction scores - how well do the first two trials predict the last. This should be the limit
    # of generalizability of the regression model: If it was perfectly fit to the first two trials this is the best
    # it should be able to do. Higher predictions should be spuriuous
    t_len = fish_to_test.shape[0]
    self_scores = np.array([np.corrcoef(ftf[:t_len] + ftf[t_len:], ftt)[0, 1] ** 2 for (ftf, ftt)
                            in zip(fish_to_fit.T, fish_to_test.T)])

    # create map of cells in fish brain colored by given score (alpha and degree of orange-ness correlates with scores)
    plot_colors = np.vstack([(0, s, s * .5, s ** 2) for s in scores])
    plot_threshold = 0.25  # minimum score (R^2) for a unit to be plotted
    to_plot = np.logical_and(scores > plot_threshold, cell_types == b"MAIN")
    to_plot = np.logical_and(to_plot, self_scores >= scores)  # enforce generalization criterion
    # top view
    fig, ax = pl.subplots()
    ax.scatter(cell_centroids[to_plot, 0], cell_centroids[to_plot, 1], s=1, color=(0, 0.5, 0.25, 0.3))
    ax.set_aspect('equal', 'datalim')
    sns.despine(fig, ax)
    fig.savefig(save_folder + "prediction_score_map_top_view.pdf", type="pdf")
    # side view
    fig, ax = pl.subplots()
    ax.scatter(cell_centroids[to_plot, 1], cell_centroids[to_plot, 2], s=1, color=(0, 0.5, 0.25, 0.3))
    ax.set_aspect('equal', 'datalim')
    sns.despine(fig, ax)
    fig.savefig(save_folder + "prediction_score_map_side_view.pdf", type="pdf")

    # plot comparison maps of clustered fish data
    to_plot = np.logical_and(zf_clusters > -1, zf_clusters < 6)
    to_plot = np.logical_and(to_plot, cell_types == b"MAIN")
    to_plot = np.logical_and(to_plot, np.random.rand(to_plot.size) > 0.25)
    # top view
    fig, ax = pl.subplots()
    ax.scatter(cell_centroids[to_plot, 0], cell_centroids[to_plot, 1], s=1, color=(0, 0.5, 0.25, 0.3))
    ax.set_aspect('equal', 'datalim')
    sns.despine(fig, ax)
    fig.savefig(save_folder + "fish_clustered_map_top_view.pdf", type="pdf")
    # side view
    fig, ax = pl.subplots()
    ax.scatter(cell_centroids[to_plot, 1], cell_centroids[to_plot, 2], s=1, color=(0, 0.5, 0.25, 0.3))
    ax.set_aspect('equal', 'datalim')
    sns.despine(fig, ax)
    fig.savefig(save_folder + "fish_clustered_map_side_view.pdf", type="pdf")

    # compute overlap of cells positive by prediction (R2 > 0.25) together with random expectations
    predicted = np.logical_and(scores > plot_threshold, self_scores >= scores)
    clustered = np.logical_and(zf_clusters > -1, zf_clusters < 6)
    # matrix:                | Prediction positive | Prediction negative
    #          Fish positive |                     |
    #          Fish negative |                     |
    overlap = np.zeros((2, 2))
    overlap[0, 0] = np.sum(np.logical_and(predicted, clustered))
    overlap[1, 0] = np.sum(np.logical_and(predicted, np.logical_not(clustered)))
    overlap[0, 1] = np.sum(np.logical_and(np.logical_not(predicted), clustered))
    overlap[1, 1] = np.sum(np.logical_and(np.logical_not(predicted), np.logical_not(clustered)))
    overlap_random = np.zeros_like(overlap)
    for shuffles in range(1000):
        np.random.shuffle(predicted)
        overlap_random[0, 0] += np.sum(np.logical_and(predicted, clustered))
        overlap_random[1, 0] += np.sum(np.logical_and(predicted, np.logical_not(clustered)))
        overlap_random[0, 1] += np.sum(np.logical_and(np.logical_not(predicted), clustered))
        overlap_random[1, 1] += np.sum(np.logical_and(np.logical_not(predicted), np.logical_not(clustered)))
    overlap_random /= 1000
    print("True overlap between prediction and clustering:")
    print(overlap.astype(int))
    print("Overlap expected by chance:")
    print(overlap_random.astype(int))
