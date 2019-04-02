#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script for panels of Figure S2 (Zebrafish network characterization)
"""


import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pl
import seaborn as sns
from mo_types import MoTypes
import core as c
import analysis as a
import h5py
from global_defs import GlobalDefs
from pandas import DataFrame
from Figure4 import mpath
from scipy.signal import convolve
from multiprocessing import Pool


# file definitions
base_path_zf = "./model_data/Adam_1e-4/sepInput_mixTrain/"
paths_512_zf = [f + '/' for f in os.listdir(base_path_zf) if "_3m512_" in f]

base_path_05Hz = "./model_data/Adam_1e-4/halfHz/"
paths_05Hz = [f+'/' for f in os.listdir(base_path_05Hz) if "_3m512_" in f]

base_path_2Hz = "./model_data/Adam_1e-4/doubleHz/"
paths_2Hz = [f+'/' for f in os.listdir(base_path_2Hz) if "_3m512_" in f]


def compute_white_noise(base_freq):
    if base_freq == 0.5:
        paths = paths_05Hz
        base = base_path_05Hz
        std = std_05Hz
        n_steps = 100000000
    elif base_freq == 1.0:
        paths = paths_512_zf
        base = base_path_zf
        std = std_zf
        n_steps = 10000000  # there are 10 times as many models
    elif base_freq == 2.0:
        paths = paths_2Hz
        base = base_path_2Hz
        std = std_2Hz
        n_steps = 50000000
    else:
        raise ValueError("Indicated base frequency has not been trained")
    behav_kernels = {}
    k_names = ["stay", "straight", "left", "right"]
    for p in paths:
        m_path = mpath(base, p)
        mdata_wn = c.ModelData(m_path)
        gpn_wn = mo.network_model()
        gpn_wn.load(mdata_wn.ModelDefinition, mdata_wn.LastCheckpoint)
        wna = mo.wn_sim(std, gpn_wn, stim_std=2)
        wna.p_move *= base_freq
        wna.bf_mult = base_freq
        wna.switch_mean = 5
        wna.switch_std = 1
        ev_path = m_path + '/evolve/generation_weights.npy'
        weights = np.load(ev_path)
        w = np.mean(weights[-1, :, :], 0)
        wna.bf_weights = w
        kernels = wna.compute_behavior_kernels(n_steps)
        for i, n in enumerate(k_names):
            if n in behav_kernels:
                behav_kernels[n].append(kernels[i])
            else:
                behav_kernels[n] = [kernels[i]]
    time = np.linspace(-4, 1, behav_kernels['straight'][0].size)
    for n in k_names:
        behav_kernels[n] = np.vstack(behav_kernels[n])
    plot_kernel = (behav_kernels["straight"] + behav_kernels["left"] + behav_kernels["right"]) / 3
    return time, plot_kernel


def unit_wn_helper(mo_type, model_path, std, nsamples):
    md_wn = c.ModelData(model_path)
    gpn_wnsim = mo_type.network_model()
    gpn_wnsim.load(md_wn.ModelDefinition, md_wn.LastCheckpoint)
    wnsim = mo_type.wn_sim(std, gpn_wnsim, stim_std=2)
    wnsim.switch_mean = 5
    wnsim.switch_std = 1
    ev_path = model_path + '/evolve/generation_weights.npy'
    ev_weights = np.load(ev_path)
    wts = np.mean(ev_weights[-1, :, :], 0)
    wnsim.bf_weights = wts
    all_triggered_units = wnsim.compute_behav_trig_activity(nsamples)
    units_straight = all_triggered_units[1]['t']  # only use units in temperature branch
    left = all_triggered_units[2]['t']
    right = all_triggered_units[3]['t']
    units_turn = [l + r for (l, r) in zip(left, right)]
    return units_straight, units_turn


if __name__ == "__main__":
    save_folder = "./DataFigures/FigureS2/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42

    std_05Hz = c.GradientData.load_standards("gd_05Hz_training_data.hdf5")
    std_2Hz = c.GradientData.load_standards("gd_2Hz_training_Data.hdf5")
    std_zf = c.GradientData.load_standards("gd_training_data.hdf5")
    ana_zf = a.Analyzer(MoTypes(False), std_zf, "sim_store.hdf5", "activity_store.hdf5")

    # load cluster data from file
    clfile = h5py.File("cluster_info.hdf5", "r")
    clust_ids_zf = np.array(clfile["clust_ids"])
    clfile.close()

    # load and interpolate temperature stimulus
    dfile = h5py.File("stimFile.hdf5", 'r')
    tsin = np.array(dfile['sine_L_H_temp'])
    x = np.arange(tsin.size)  # stored at 20 Hz !
    xinterp = np.linspace(0, tsin.size, tsin.size * GlobalDefs.frame_rate // 20)
    temperature = np.interp(xinterp, x, tsin)
    dfile.close()

    # get activity data
    all_ids_zf = []
    all_cells_zf = []
    for i, p in enumerate(paths_512_zf):
        cell_res, ids = ana_zf.temperature_activity(mpath(base_path_zf, p), temperature, i)
        all_ids_zf.append(ids)
        all_cells_zf.append(cell_res)
    all_ids_zf = np.hstack(all_ids_zf)
    all_cells_zf = np.hstack(all_cells_zf)

    # convolve activity with nuclear gcamp calcium kernel
    tau_on = 1.4  # seconds
    tau_on *= GlobalDefs.frame_rate  # in frames
    tau_off = 2  # seconds
    tau_off *= GlobalDefs.frame_rate  # in frames
    kframes = np.arange(10 * GlobalDefs.frame_rate)  # 10 s long kernel
    kernel = 2 ** (-kframes / tau_off) * (1 - 2 ** (-kframes / tau_on))
    kernel = kernel / kernel.sum()
    # convolve with our kernel
    for i in range(all_cells_zf.shape[1]):
        all_cells_zf[:, i] = convolve(all_cells_zf[:, i], kernel, mode='full')[:all_cells_zf.shape[0]]

    # plot colors
    pal = sns.color_palette()  # the default matplotlib color cycle
    plot_cols_zf = {0: (0.6, 0.6, 0.6), 1: pal[2], 2: (102 / 255, 45 / 255, 145 / 255), 3: pal[0], 4: pal[3], 5: pal[1],
                    6: pal[5], 7: (76 / 255, 153 / 255, 153 / 255), -1: (0.6, 0.6, 0.6)}

    # panel - all cluster activities, sorted into ON and OFF types
    n_regs = np.unique(clust_ids_zf).size - 1
    cluster_acts = np.zeros((all_cells_zf.shape[0] // 3, n_regs))
    for i in range(n_regs):
        cluster_acts[:, i] = np.mean(a.trial_average(all_cells_zf[:, clust_ids_zf == i], 3), 1)
    on_count = 0
    off_count = 0
    fig, (axes_on, axes_off) = pl.subplots(ncols=2, nrows=2, sharey=True, sharex=True)
    time = np.arange(cluster_acts.shape[0]) / GlobalDefs.frame_rate
    for i in range(n_regs):
        act = cluster_acts[:, i]
        if np.corrcoef(act, temperature[:act.size])[0, 1] < 0:
            ax_off = axes_off[0] if off_count < 2 else axes_off[1]
            ax_off.plot(time, cluster_acts[:, i], color=plot_cols_zf[i])
            off_count += 1
        else:
            ax_on = axes_on[0] if on_count < 2 else axes_on[1]
            ax_on.plot(time, cluster_acts[:, i], color=plot_cols_zf[i])
            on_count += 1
    axes_off[0].set_xticks([0, 30, 60, 90, 120, 150])
    axes_off[1].set_xticks([0, 30, 60, 90, 120, 150])
    axes_off[0].set_xlabel("Time [s]")
    axes_off[1].set_xlabel("Time [s]")
    axes_on[0].set_ylabel("Cluster average activation")
    axes_off[0].set_ylabel("Cluster average activation")
    sns.despine()
    fig.tight_layout()
    fig.savefig(save_folder + "zf_all_cluster_averages.pdf", type="pdf")

    # panel - average type counts in temperature branch for each cluster
    cl_type_d = {"Fraction": [], "net_id": [], "Cluster ID": [], "Layer": []}
    for i in range(len(paths_512_zf)):
        for j in range(-1, n_regs):
            for k in range(2):
                lay_clust_ids = clust_ids_zf[np.logical_and(all_ids_zf[0, :] == i, all_ids_zf[1, :] == k)]
                cl_type_d["Fraction"].append(np.sum(lay_clust_ids == j) / 512)
                cl_type_d["net_id"].append(i)
                cl_type_d["Cluster ID"].append(j)
                cl_type_d["Layer"].append(k)
    cl_type_df = DataFrame(cl_type_d)
    fig, (ax_0, ax_1) = pl.subplots(nrows=2, sharex=True)
    sns.barplot("Cluster ID", "Fraction", data=cl_type_df[cl_type_df["Layer"] == 0], order=list(range(n_regs)) + [-1],
                ci=68, ax=ax_0, palette=plot_cols_zf)
    sns.barplot("Cluster ID", "Fraction", data=cl_type_df[cl_type_df["Layer"] == 1], order=list(range(n_regs)) + [-1],
                ci=68, ax=ax_1, palette=plot_cols_zf)
    ax_0.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
    ax_1.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
    sns.despine(fig)
    fig.savefig(save_folder + "zf_all_cluster_counts.pdf", type="pdf")

    # panel for ON-OFF type search in zebrafish
    on_off = 7
    # load fish activity data
    dfile = h5py.File('zebrafish_brain_data.hdf5', 'r')
    all_activity = np.array(dfile['all_activity'])
    no_nan_aa = np.array(dfile['no_nan_aa'])
    tf_centroids = np.array(dfile['tf_centroids'])[no_nan_aa, :]
    frame_times = np.arange(all_activity.shape[1]) / 5
    dfile.close()
    dfile = h5py.File("stack_types.hdf5", 'r')
    stack_types = np.array(dfile["stack_types"])[no_nan_aa]
    dfile.close()
    network_times = np.arange(all_cells_zf.shape[0]) / GlobalDefs.frame_rate
    on_off_regressor = np.mean(all_cells_zf[:, clust_ids_zf == on_off], 1)
    on_off_regressor = np.interp(frame_times, network_times, on_off_regressor)
    on_off_corrs = np.zeros(all_activity.shape[0])
    for i, act in enumerate(all_activity):
        on_off_corrs[i] = np.corrcoef(act, on_off_regressor)[0, 1]
    on_off_fish = a.trial_average(all_activity[on_off_corrs > 0.6, :].T, 3).T
    F0 = np.mean(on_off_fish[:, :30 * 5], 1, keepdims=True)
    on_off_fish = (on_off_fish - F0) / F0
    on_off_netw = a.trial_average(on_off_regressor[:, None], 3).ravel()
    fish_trial_time = np.arange(on_off_fish.shape[1]) / 5
    fig, (ax_net, ax_fish) = pl.subplots(nrows=2, sharex=True)
    ax_net.plot(fish_trial_time, on_off_netw, color=(76 / 255, 153 / 255, 153 / 255))
    sns.tsplot(on_off_fish, fish_trial_time, color=(76 / 255, 153 / 255, 153 / 255), ax=ax_fish)
    ax_fish.set_xlabel("Time [s]")
    ax_fish.set_ylabel("Activity [dF/F]")
    ax_net.set_ylabel("Activation")
    ax_fish.set_xticks([0, 30, 60, 90, 120, 150])
    ax_net.set_xticks([0, 30, 60, 90, 120, 150])
    sns.despine(fig)
    fig.savefig(save_folder + "on_off_type_activity.pdf", type="pdf")

    raise Exception("Skipping white noise analysis to save time")
    # panels - behavior triggered cluster averages during white noise
    mo = MoTypes(False)
    all_units_straight = []
    all_units_turn = []
    for i, p in enumerate(paths_512_zf):
        m_path = mpath(base_path_zf, p)
        # NOTE: Due to high memory requirements unfortunately need to accumulate this all serially...
        straight, turn = unit_wn_helper(mo, m_path, std_zf, 250000)
        for rep in range(19):
            st, tu = unit_wn_helper(mo, m_path, std_zf, 250000)
            for lix in range(len(straight)):
                straight[lix] = (straight[lix] + st[lix])/2
                turn[lix] = (turn[lix] + tu[lix])/2
        all_units_straight += straight
        all_units_turn += turn
        print("Network {0} of {1} completed".format(i+1, len(paths_512_zf)))
    all_units_straight = np.hstack(all_units_straight)
    all_units_turn = np.hstack(all_units_turn)
    # plot kernels for our unit types - NOTE: Assignments below are retrieved via zf_ann_correspondence.py except
    # for 'integrating OFF' which is determined via brain wide unit correlations (also see Figure2)
    fast_on_like = 4
    slow_on_like = 5
    fast_off_like = 1
    slow_off_like = 3
    int_off = 2
    kernel_time = np.linspace(-4, 1, all_units_straight.shape[0])
    fig, ax = pl.subplots()
    sns.tsplot(all_units_turn[:, clust_ids_zf == fast_on_like].T, kernel_time, n_boot=1000, color="C0", ax=ax)
    sns.tsplot(all_units_straight[:, clust_ids_zf == fast_on_like].T, kernel_time, n_boot=1000, color="C1", ax=ax)
    ax.plot([kernel_time.min(), kernel_time.max()], [0, 0], 'k--', lw=0.25)
    ax.plot([0, 0], [-0.001, 0.001], 'k--', lw=0.25)
    ax.set_ylabel("Activation")
    ax.set_xlabel("Time around bout [s]")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "behav_triggered_FastON.pdf", type="pdf")
    fig, ax = pl.subplots()
    sns.tsplot(all_units_turn[:, clust_ids_zf == slow_on_like].T, kernel_time, n_boot=1000, color="C0", ax=ax)
    sns.tsplot(all_units_straight[:, clust_ids_zf == slow_on_like].T, kernel_time, n_boot=1000, color="C1", ax=ax)
    ax.plot([kernel_time.min(), kernel_time.max()], [0, 0], 'k--', lw=0.25)
    ax.plot([0, 0], [-0.001, 0.001], 'k--', lw=0.25)
    ax.set_ylabel("Activation")
    ax.set_xlabel("Time around bout [s]")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "behav_triggered_SlowON.pdf", type="pdf")
    fig, ax = pl.subplots()
    sns.tsplot(all_units_turn[:, clust_ids_zf == fast_off_like].T, kernel_time, n_boot=1000, color="C0", ax=ax)
    sns.tsplot(all_units_straight[:, clust_ids_zf == fast_off_like].T, kernel_time, n_boot=1000, color="C1", ax=ax)
    ax.plot([kernel_time.min(), kernel_time.max()], [0, 0], 'k--', lw=0.25)
    ax.plot([0, 0], [-0.001, 0.001], 'k--', lw=0.25)
    ax.set_ylabel("Activation")
    ax.set_xlabel("Time around bout [s]")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "behav_triggered_FastOFF.pdf", type="pdf")
    fig, ax = pl.subplots()
    sns.tsplot(all_units_turn[:, clust_ids_zf == slow_off_like].T, kernel_time, n_boot=1000, color="C0", ax=ax)
    sns.tsplot(all_units_straight[:, clust_ids_zf == slow_off_like].T, kernel_time, n_boot=1000, color="C1", ax=ax)
    ax.plot([kernel_time.min(), kernel_time.max()], [0, 0], 'k--', lw=0.25)
    ax.plot([0, 0], [-0.001, 0.001], 'k--', lw=0.25)
    ax.set_ylabel("Activation")
    ax.set_xlabel("Time around bout [s]")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "behav_triggered_SlowOFF.pdf", type="pdf")
    fig, ax = pl.subplots()
    sns.tsplot(all_units_turn[:, clust_ids_zf == int_off].T, kernel_time, n_boot=1000, color="C0", ax=ax)
    sns.tsplot(all_units_straight[:, clust_ids_zf == int_off].T, kernel_time, n_boot=1000, color="C1", ax=ax)
    ax.plot([kernel_time.min(), kernel_time.max()], [0, 0], 'k--', lw=0.25)
    ax.plot([0, 0], [-0.001, 0.001], 'k--', lw=0.25)
    ax.set_ylabel("Activation")
    ax.set_xlabel("Time around bout [s]")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "behav_triggered_IntOFF.pdf", type="pdf")

    # panel 1 - white noise analysis on naive networks
    behav_kernels = {}
    k_names = ["stay", "straight", "left", "right"]
    for p in paths_512_zf:
        m_path = mpath(base_path_zf, p)
        mdata_wn = c.ModelData(m_path)
        gpn_wn = mo.network_model()
        gpn_wn.load(mdata_wn.ModelDefinition, mdata_wn.FirstCheckpoint)
        wna = mo.wn_sim(std_zf, gpn_wn, stim_std=2)
        wna.switch_mean = 5
        wna.switch_std = 1
        kernels = wna.compute_behavior_kernels(10000000)
        for j, n in enumerate(k_names):
            if n in behav_kernels:
                behav_kernels[n].append(kernels[j])
            else:
                behav_kernels[n] = [kernels[j]]
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
    fig.savefig(save_folder+"zf_naive_white_noise_kernels.pdf", type="pdf")

    plot_kernel_05Hz = compute_white_noise(0.5)[1]
    plot_kernel_1Hz = compute_white_noise(1.0)[1]
    kernel_time, plot_kernel_2Hz = compute_white_noise(2.0)

    m_05Hz = np.mean(plot_kernel_05Hz, 0)
    m_1Hz = np.mean(plot_kernel_1Hz, 0)
    m_2Hz = np.mean(plot_kernel_2Hz, 0)

    #  panel - white noise analysis comparison
    fig, ax = pl.subplots()
    ax.plot(kernel_time, m_05Hz, label="0.5 Hz")
    ax.plot(kernel_time, m_1Hz, label="1.0 Hz")
    ax.plot(kernel_time, m_2Hz, label="2.0 Hz")
    ax.plot([kernel_time.min(), kernel_time.max()], [0, 0], 'k--', lw=0.25)
    ax.plot([0, 0], [-0.1, 0.3], 'k--', lw=0.25)
    ax.set_ylabel("Normalized filter")
    ax.set_xlabel("Time around bout [s]")
    ax.legend()
    sns.despine(fig, ax)
    fig.savefig(save_folder + "wn_kernels_moveFreq_compare.pdf", type="pdf")
