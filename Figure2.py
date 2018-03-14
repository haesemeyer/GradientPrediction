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
        all_cells[:, i] = convolve(all_cells[:, i], kernel, method='full')[:all_cells.shape[0]]
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
