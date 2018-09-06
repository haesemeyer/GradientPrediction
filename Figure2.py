#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script for panels of Figure 2 (Zebrafish model behavior and neuron responses)
"""


import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pl
import seaborn as sns
from Figure1 import mpath
from mo_types import MoTypes
import core as c
import analysis as a
import h5py
from global_defs import GlobalDefs
from scipy.signal import convolve


# file definitions
base_path = "./model_data/Adam_1e-4/sepInput_mixTrain/"

paths_512 = [f+'/' for f in os.listdir(base_path) if "_3m512_" in f]


if __name__ == "__main__":
    save_folder = "./DataFigures/Figure2/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42
    mo = MoTypes(False)
    std = c.GradientData.load_standards("gd_training_data.hdf5")
    ana = a.Analyzer(mo, std, "sim_store.hdf5", "activity_store.hdf5")

    # for fish-like clusters - their indices
    fast_on_like = 4
    slow_on_like = 5
    fast_off_like = 1
    slow_off_like = 3

    # for clusters we identify in fish - their indices
    int_off = 2

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

    # get activity data
    all_cells = []
    all_ids = []
    for i, p in enumerate(paths_512):
        cell_res, ids = ana.temperature_activity(mpath(p), temperature, i)
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
    f_on_data = a.trial_average(all_cells[:, clust_ids == fast_on_like], 3).T
    s_on_data = a.trial_average(all_cells[:, clust_ids == slow_on_like], 3).T
    f_off_data = a.trial_average(all_cells[:, clust_ids == fast_off_like], 3).T
    s_off_data = a.trial_average(all_cells[:, clust_ids == slow_off_like], 3).T
    trial_time = np.arange(f_on_data.shape[1]) / GlobalDefs.frame_rate

    # second panel - fish-like on types
    fig, ax = pl.subplots()
    sns.tsplot(f_on_data, trial_time, n_boot=1000, color="C3")
    sns.tsplot(s_on_data, trial_time, n_boot=1000, color="C1")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Activation [AU]")
    ax.set_xticks([0, 30, 60, 90, 120, 150])
    sns.despine(fig, ax)
    fig.savefig(save_folder + "fishlike_on_types.pdf", type="pdf")

    # third panel - fish-like off types
    fig, ax = pl.subplots()
    sns.tsplot(f_off_data, trial_time, n_boot=1000, color="C2")
    sns.tsplot(s_off_data, trial_time, n_boot=1000, color="C0")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Activation [AU]")
    ax.set_xticks([0, 30, 60, 90, 120, 150])
    sns.despine(fig, ax)
    fig.savefig(save_folder + "fishlike_off_types.pdf", type="pdf")

    # fourth panel - fish-response and location corresponding to integrating off type
    # load fish activity data
    dfile = h5py.File('H:/ClusterLocations_170327_clustByMaxCorr/datafile_170327.hdf5', 'r')
    all_activity = np.array(dfile['all_activity'])
    no_nan_aa = np.array(dfile['no_nan_aa'])
    tf_centroids = np.array(dfile['tf_centroids'])[no_nan_aa, :]
    frame_times = np.arange(all_activity.shape[1]) / 5
    dfile.close()
    dfile = h5py.File("stack_types.hdf5", 'r')
    stack_types = np.array(dfile["stack_types"])[no_nan_aa]
    dfile.close()
    network_times = np.arange(all_cells.shape[0]) / GlobalDefs.frame_rate
    int_off_regressor = np.mean(all_cells[:, clust_ids == int_off], 1)
    int_off_regressor = np.interp(frame_times, network_times, int_off_regressor)
    int_off_corrs = np.zeros(all_activity.shape[0])
    for i, act in enumerate(all_activity):
        int_off_corrs[i] = np.corrcoef(act, int_off_regressor)[0, 1]
    int_off_fish = a.trial_average(all_activity[int_off_corrs > 0.6, :].T, 3).T
    F0 = np.mean(int_off_fish[:, :30*5], 1, keepdims=True)
    int_off_fish = (int_off_fish-F0) / F0
    int_off_netw = a.trial_average(int_off_regressor[:, None], 3).ravel()
    fish_trial_time = np.arange(int_off_fish.shape[1]) / 5
    fig, (ax_net, ax_fish) = pl.subplots(nrows=2, sharex=True)
    ax_net.plot(fish_trial_time, int_off_netw, color=(76 / 255, 153 / 255, 153 / 255))
    sns.tsplot(int_off_fish, fish_trial_time, color=(76 / 255, 153 / 255, 153 / 255), ax=ax_fish, err_style="ci_band")
    ax_fish.set_xlabel("Time [s]")
    ax_fish.set_ylabel("Activity [dF/F]")
    ax_net.set_ylabel("Activation")
    ax_fish.set_xticks([0, 30, 60, 90, 120, 150])
    ax_net.set_xticks([0, 30, 60, 90, 120, 150])
    sns.despine(fig)
    fig.savefig(save_folder + "integrating_off_type_activity.pdf", type="pdf")
    # cell location dorsal view
    int_off_cents = tf_centroids[np.logical_and(int_off_corrs > 0.6, stack_types == b"MAIN"), :]
    rand_cents = tf_centroids[np.logical_and(np.random.rand(tf_centroids.shape[0]) < 0.002, stack_types == b"MAIN"), :]
    fig, ax = pl.subplots()
    ax.scatter(rand_cents[:, 0], rand_cents[:, 1], s=1, color='k', alpha=0.1)
    ax.scatter(int_off_cents[:, 0], int_off_cents[:, 1], s=2, color=(76 / 255, 153 / 255, 153 / 255))
    ax.set_aspect('equal', 'datalim')
    sns.despine(fig, ax)
    fig.savefig(save_folder + "integrating_off_type_top_view.pdf", type="pdf")
    # cell location lateral view
    fig, ax = pl.subplots()
    ax.scatter(rand_cents[:, 1], rand_cents[:, 2], s=1, color='k', alpha=0.1)
    ax.scatter(int_off_cents[:, 1], int_off_cents[:, 2], s=2, color=(76 / 255, 153 / 255, 153 / 255))
    ax.set_aspect('equal', 'datalim')
    sns.despine(fig, ax)
    fig.savefig(save_folder + "integrating_off_type_side_view.pdf", type="pdf")

    # fifth panel - input connectivity into second layer of t branch
    conn_mat = np.zeros((8, 8, len(paths_512)))
    for i, p in enumerate(paths_512):
        model_cids = clust_ids[all_ids[0, :] == i]
        layers_ids = all_ids[1, :][all_ids[0, :] == i]
        m_path = mpath(p)
        mdata = c.ModelData(m_path)
        gpn = mo.network_model()
        gpn.load(mdata.ModelDefinition, mdata.LastCheckpoint)
        conn_mat[:, :, i] = gpn.parse_layer_input_by_cluster('t', 1, model_cids[layers_ids == 0],
                                                             model_cids[layers_ids == 1])
    # reordered version of conn_mat based on known types
    cm_order = [fast_on_like, slow_on_like, fast_off_like, slow_off_like, int_off, 0, 6, 7]
    cm_reordered = conn_mat[:, cm_order, :]
    cm_reordered = cm_reordered[cm_order, :, :]
    m = np.mean(cm_reordered, 2)
    s = np.std(cm_reordered, 2)
    cross_0 = np.sign((m+s) * (m-s)) <= 0
    m[cross_0] = 0
    s[cross_0] = 0
    fig, axes = pl.subplots(nrows=5, sharex=True, sharey=True)
    for i in range(5):
        axes[i].bar(np.arange(8), m[:, i], width=[.8]*5+[.3]*3)
        axes[i].errorbar(np.arange(8), m[:, i], s[:, i], color='k', fmt='none')
    axes[-1].set_xticks(np.arange(8))
    axes[-1].set_xticklabels(["Fast ON", "Slow ON", "Fast OFF", "Slow OFF", "Int. OFF", "O", "O", "O"])
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(save_folder + "avg_connectivity_weights.pdf", type="pdf")

    raise Exception("Skipping white noise computation to save time")
    # first panel - white noise analysis of generated behavior
    behav_kernels = {}
    k_names = ["stay", "straight", "left", "right"]
    for p in paths_512:
        m_path = mpath(p)
        mdata_wn = c.ModelData(m_path)
        gpn_wn = mo.network_model()
        gpn_wn.load(mdata_wn.ModelDefinition, mdata_wn.LastCheckpoint)
        wna = mo.wn_sim(std, gpn_wn, stim_std=2)
        wna.switch_mean = 5
        wna.switch_std = 1
        ev_path = m_path + '/evolve/generation_weights.npy'
        weights = np.load(ev_path)
        w = np.mean(weights[-1, :, :], 0)
        wna.bf_weights = w
        kernels = wna.compute_behavior_kernels(10000000)
        for i, n in enumerate(k_names):
            if n in behav_kernels:
                behav_kernels[n].append(kernels[i])
            else:
                behav_kernels[n] = [kernels[i]]
    kernel_time = np.linspace(-4, 1, behav_kernels['straight'][0].size)
    for n in k_names:
        behav_kernels[n] = np.vstack(behav_kernels[n])
    plot_kernels = {"straight": behav_kernels["straight"], "turn": (behav_kernels["left"] + behav_kernels["right"])/2}
    fig, ax = pl.subplots()
    for i, n in enumerate(plot_kernels):
        sns.tsplot(plot_kernels[n], kernel_time, n_boot=1000, color="C{0}".format(i), ax=ax, condition=n)
    ax.plot([kernel_time.min(), kernel_time.max()], [0, 0], 'k--', lw=0.25)
    ax.plot([0, 0], [-0.1, 0.2], 'k--', lw=0.25)
    ax.set_ylabel("Filter kernel")
    ax.set_xlabel("Time around bout [s]")
    ax.legend()
    sns.despine(fig, ax)
    fig.savefig(save_folder+"white_noise_kernels.pdf", type="pdf")
