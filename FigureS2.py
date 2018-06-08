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
import pickle
from data_stores import SimulationStore, ActivityStore
from Figure4 import mpath
from scipy.signal import convolve


# file definitions
base_path_zf = "./model_data/Adam_1e-4/sepInput_mixTrain/"
paths_512_zf = [f + '/' for f in os.listdir(base_path_zf) if "_3m512_" in f]


if __name__ == "__main__":
    save_folder = "./DataFigures/FigureS2/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42

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
        all_cells_zf[:, i] = convolve(all_cells_zf[:, i], kernel, method='full')[:all_cells_zf.shape[0]]

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
    cl_type_d = {"Fraction": [], "net_id": [], "Cluster ID": []}
    for i in range(len(paths_512_zf)):
        net_clust_ids = clust_ids_zf[all_ids_zf[0, :] == i]
        for j in range(-1, n_regs):
            cl_type_d["Fraction"].append(np.sum(net_clust_ids == j) / 1024)
            cl_type_d["net_id"].append(i)
            cl_type_d["Cluster ID"].append(j)
    cl_type_df = DataFrame(cl_type_d)
    fig, ax = pl.subplots()
    sns.barplot("Cluster ID", "Fraction", data=cl_type_df, order=list(range(n_regs)) + [-1], ci=68, ax=ax,
                palette=plot_cols_zf)
    ax.set_yticks([0, 0.05, 0.1, 0.15])
    sns.despine(fig, ax)
    fig.savefig(save_folder + "zf_all_cluster_counts.pdf", type="pdf")

    raise Exception("Skipping white noise analysis to save time")
    # panel 1 - white noise analysis on naive networks
    mo = MoTypes(False)
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
