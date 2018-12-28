#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script for quantitative correspondence between zebrafish and neural network activity clusters
"""

import numpy as np
from scipy.signal import convolve
import matplotlib.pyplot as pl
import matplotlib as mpl
import seaborn as sns
import h5py
from typing import Dict
import pickle
import os
from global_defs import GlobalDefs
import analysis as a
import core as c
from mo_types import MoTypes
from Figure4 import mpath


class RegionResults:
    """
    This is an exact copy from analyzeSensMotor.py of ImagingAnalysis
    """
    def __init__(self, name, activities, membership, regressors, original_labels, region_indices):
        self.name = name
        self.region_acts = activities
        self.region_mem = membership
        self.regressors = regressors
        self.regs_clust_labels = original_labels
        self.region_indices = region_indices
        self.full_averages = None


# file definitions
base_path = "./model_data/Adam_1e-4/sepInput_mixTrain/"
paths_512 = [f+'/' for f in os.listdir(base_path) if "_3m512_" in f]


def create_corr_mat(network, zfish, times, start_time, end_time=None):
    corr_mat = np.full((network.shape[1], zfish.shape[1]), np.nan)
    if end_time is None or end_time < start_time:
        valid = times >= start_time
    else:
        valid = np.logical_and(times <= end_time, times >= start_time)
    for n in range(corr_mat.shape[0]):
        net_act = network[valid, n]
        for z in range(corr_mat.shape[1]):
            zf_reg = zfish[valid, z]
            corr_mat[n, z] = np.corrcoef(net_act, zf_reg)[0, 1]
    return corr_mat


def greedy_max_clust(corr_mat, threshold, col_names):
    """
    Tries to find best correlated row above threshold for each column giving preference to making a match
    for each column even if this requires picking a worse match in another column
    :param corr_mat: The pairwise correlations
    :param threshold: The minimal correlation to consider
    :param col_names: The names of the columns
    :return: Dictionary with rows as keys and matched column names as values
    """
    col_matches = np.full(corr_mat.shape[1], -2)
    work_mat = corr_mat.copy()
    work_mat[corr_mat < threshold] = 0
    first_run = True
    while np.any(col_matches == -2):
        for col in range(corr_mat.shape[1]):
            if col_matches[col] > -2:
                continue
            if np.all(work_mat[:, col] == 0):
                # no possible assignment - mark as completed but un-assigned
                col_matches[col] = -1
                continue
            if np.sum(work_mat[:, col] > 0) == 1:
                # if this is the only choice, assign and mark that row as used
                col_matches[col] = np.argmax(work_mat[:, col])
                work_mat[col_matches[col], :] = 0
                continue
            if not first_run:
                col_matches[col] = np.argmax(work_mat[:, col])
                work_mat[col_matches[col], :] = 0
        # indicate that all "loners" have already been assigned
        first_run = False
    # recode column matches into row matches
    row_matches = np.full(corr_mat.shape[0], -1)
    for ix, cm in enumerate(col_matches):
        row_matches[cm] = ix
    return {ix: col_names[row_matches[ix]] if row_matches[ix] != -1 else ix for ix in range(corr_mat.shape[0])}


if __name__ == "__main__":
    save_folder = "./DataFigures/ZF_ANN_Correspondence/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42

    mo = MoTypes(False)
    std = c.GradientData.load_standards("gd_training_data.hdf5")
    ana = a.Analyzer(mo, std, "sim_store.hdf5", "activity_store.hdf5")

    # load zebrafish region results and create Rh56 regressor matrix for FastON, SlowON, FastOFF, SlowOFF
    result_labels = ["Rh6"]
    region_results = {}  # type: Dict[str, RegionResults]
    analysis_file = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/regiondata.hdf5', 'r')
    for rl in result_labels:
        region_results[rl] = pickle.loads(np.array(analysis_file[rl]))
    analysis_file.close()
    rh_56_calcium = region_results["Rh6"].regressors[:, :-1]
    # the names of these regressors according to Haesemeyer et al., 2018
    reg_names = ["Fast ON", "Slow ON", "Fast OFF", "Slow OFF"]

    # load and interpolate temperature stimulus
    dfile = h5py.File("stimFile.hdf5", 'r')
    tsin = np.array(dfile['sine_L_H_temp'])
    x = np.arange(tsin.size)  # stored at 20 Hz !
    xinterp = np.linspace(0, tsin.size, tsin.size * GlobalDefs.frame_rate // 20)
    temperature = np.interp(xinterp, x, tsin)
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

    # load activity clusters from file or create if necessary
    clust_ids = a.cluster_activity(8, all_cells, "cluster_info.hdf5")[0]

    # create ANN cluster centroid matrix
    ann_cluster_centroids = np.zeros((all_cells.shape[0]//3, 8))
    for i in range(8):
        centroid = np.mean(all_cells[:, clust_ids == i], 1)
        ann_cluster_centroids[:, i] = a.trial_average(centroid[:, None], 3).ravel()

    # interpolate fish calcium data to network time base
    ca_time = np.linspace(0, 165, rh_56_calcium.shape[0])
    net_time = np.linspace(0, 165, ann_cluster_centroids.shape[0])
    zf_cluster_centroids = np.zeros((net_time.size, rh_56_calcium.shape[1]))
    for i in range(rh_56_calcium.shape[1]):
        zf_cluster_centroids[:, i] = np.interp(net_time, ca_time, rh_56_calcium[:, i])

    # perform all pairwise correlations between the network and zebrafish units during sine stimulus phase
    cm_sine = create_corr_mat(ann_cluster_centroids, zf_cluster_centroids, net_time, 60, 105)
    assignment = greedy_max_clust(cm_sine, 0.6, reg_names)
    assign_labels = [assignment[k] for k in range(cm_sine.shape[0])]

    # plot correlation matrix
    fig, ax = pl.subplots()
    sns.heatmap(cm_sine, vmin=-1, vmax=1, center=0, annot=True, xticklabels=reg_names, yticklabels=assign_labels, ax=ax,
                cmap="RdBu_r")
    ax.set_xlabel("Zebrafish cell types")
    ax.set_ylabel("ANN clusters")
    fig.tight_layout()
    fig.savefig(save_folder + "ZFish_ANN_Correspondence.pdf", type="pdf")
