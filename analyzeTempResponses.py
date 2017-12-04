#  Copyright 2017 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script to cluster and analyze temperature responses of individual network neurons
"""


import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
import sys
import matplotlib as mpl
from core import ModelData, GpNetworkModel, GradientData, ca_convolve, FRAME_RATE
import h5py
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import SpectralClustering


def trial_average(mat, n_trials):
    """
    Computes the trial average of each trace in mat
    :param mat: n-timepoints x m-cells matrix of traces
    :param n_trials: The number of trials
    :return: Trial average activity of shape (n-timepoints/n_trials) x m-cells
    """
    if mat.shape[0] % n_trials != 0:
        raise ValueError("Number of timepoints can't be divided into select number of trials")
    t_length = mat.shape[0] // n_trials
    return np.mean(mat.reshape((n_trials, t_length, mat.shape[1])), 0)


def get_cell_responses(temp, standards):
    """
    Loads a model and computes the temperature response of all neurons returning response matrix
    :return: n-timepoints x m-neurons matrix of responses
    """
    print("Select model directory")
    root = tk.Tk()
    root.update()
    root.withdraw()
    model_dir = filedialog.askdirectory(title="Select directory with model checkpoints", initialdir="./model_data/")
    mdata = ModelData(model_dir)
    root.update()
    # create our model and load from last checkpoint
    gpn = GpNetworkModel()
    gpn.load(mdata.ModelDefinition, mdata.LastCheckpoint)
    # prepend lead-in to stimulus
    lead_in = np.full(gpn.input_dims[2] - 1, np.mean(temp[:10]))
    temp = np.r_[lead_in, temp]
    activities = gpn.unit_stimulus_responses(temp, None, None, standards)
    return np.hstack(activities['t']) if 't' in activities else np.hstack(activities['m'])


def cluster_responses(response_mat, n_clusters, corr_cut=0.6):
    """
    Clusters the neuron responses using spectral clustering
    :param response_mat: The response matrix with all neuron responses
    :param n_clusters: The desired number of clusters
    :param corr_cut: The correlation cutoff to consider a given neuron to be part of a cluster
    :return:
        [0]: The cluster ids
        [1]: 3D embedding coordinates for plotting
    """
    # create trial average
    response_mat = trial_average(response_mat, 3)
    # compute pairwise correlations
    pw_corrs = np.corrcoef(response_mat.T)
    pw_corrs[np.isnan(pw_corrs)] = 0
    pw_corrs[pw_corrs < 0.2] = 0
    # perform spectral clustering
    spec_clust = SpectralClustering(n_clusters, affinity="precomputed")
    clust_ids = spec_clust.fit_predict(pw_corrs)
    spec_emb = SpectralEmbedding(3, affinity="precomputed")
    coords = spec_emb.fit_transform(pw_corrs)
    # use correlation to cluster centroids to determine final cluster membership
    regressors = np.zeros((response_mat.shape[0], n_clusters))
    for i in range(n_clusters):
        regressors[:, i] = np.mean(response_mat[:, clust_ids == i], 1)
    for i in range(response_mat.shape[1]):
        max_ix = -1
        max_corr = 0
        for j in range(n_clusters):
            c = np.corrcoef(response_mat[:, i], regressors[:, j])[0, 1]
            if c >= corr_cut and c > max_corr:
                max_ix = j
                max_corr = c
            clust_ids[i] = max_ix
    return clust_ids, coords


if __name__ == "__main__":
    if sys.platform == "darwin" and "Tk" not in mpl.get_backend():
        print("On OSX tkinter likely does not work properly if matplotlib uses a backend that is not TkAgg!")
        print("If using ipython activate TkAgg backend with '%matplotlib tk' and retry.")
        sys.exit(1)
    # load training data to obtain temperature scaling
    try:
        std = GradientData.load_standards("gd_training_data.hdf5")
    except IOError:
        print("No standards found attempting to load full training data")
        train_data = GradientData.load("gd_training_data.hdf5")
        std = train_data.standards
    # load and interpolate temperature stimulus
    dfile = h5py.File("stimFile.hdf5", 'r')
    tsin = np.array(dfile['sine_L_H_temp'])
    x = np.arange(tsin.size)  # stored at 20 Hz !
    xinterp = np.linspace(0, tsin.size, tsin.size * FRAME_RATE // 20)
    temp = np.interp(xinterp, x, tsin)
    dfile.close()
    all_cells = get_cell_responses(temp, std)
    # convolve with 3s calcium kernel
    for i in range(all_cells.shape[1]):
        all_cells[:, i] = ca_convolve(all_cells[:, i], 3.0, FRAME_RATE)
    n_regs = 8
    clust_ids, coords = cluster_responses(all_cells, n_regs)
    # trial average the "cells"
    all_cells = trial_average(all_cells, 3)

    fig = pl.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(n_regs):
        ax.scatter(coords[clust_ids == i, 0], coords[clust_ids == i, 1], coords[clust_ids == i, 2], s=5)

    fig, (ax_on, ax_off) = pl.subplots(ncols=2)
    time = np.arange(all_cells.shape[0]) / FRAME_RATE
    for i in range(n_regs):
        act = np.mean(all_cells[:, clust_ids == i], 1)
        if np.corrcoef(act, temp[:act.size])[0, 1] < 0:
            ax_off.plot(time, act)
        else:
            ax_on.plot(time, act)
    ax_off.set_xlabel("Time [s]")
    ax_off.set_ylabel("Cluster average activity")
    ax_on.set_xlabel("Time [s]")
    ax_on.set_ylabel("Cluster average activity")
    sns.despine()
    fig.tight_layout()
