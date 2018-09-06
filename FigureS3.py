#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script for panels of Figure S3 (Zebrafish tanh network characterization)
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
from sklearn.decomposition import PCA


# file definitions
base_path_th = "./model_data/Adam_1e-4/tanh/"
paths_512_th = [f + '/' for f in os.listdir(base_path_th) if "_3m512_" in f]

base_path_zf = "./model_data/Adam_1e-4/sepInput_mixTrain/"
paths_512_zf = [f + '/' for f in os.listdir(base_path_zf) if "_3m512_" in f]


def test_loss(base_path, path):
    fname = base_path + path + "losses.hdf5"
    lossfile = h5py.File(fname, "r")
    test_losses = np.array(lossfile["test_losses"])
    rank_errors = np.array(lossfile["test_rank_errors"])
    timepoints = np.array(lossfile["test_eval"])
    return timepoints, test_losses, rank_errors


if __name__ == "__main__":
    save_folder = "./DataFigures/FigureS3/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42

    # first panel - log squared error progression over training
    test_time = test_loss(base_path_th, paths_512_th[0])[0]
    test_512 = np.vstack([test_loss(base_path_th, lp)[1] for lp in paths_512_th])
    test_relu = np.vstack([test_loss(base_path_zf, lp)[1] for lp in paths_512_zf])
    fig, ax = pl.subplots()
    sns.tsplot(np.log10(test_512), test_time, ax=ax, color="C1", n_boot=1000, condition="Tanh")
    ax.plot(test_time, np.mean(np.log10(test_relu), 0), 'k', lw=0.25, label="Relu")
    epoch_times = np.linspace(0, test_time.max(), 10, endpoint=False)
    for e in epoch_times:
        ax.plot([e, e], [-1.2, .4], 'k--', lw=0.25)
    ax.set_ylabel("log(Squared test error)")
    ax.set_xlabel("Training step")
    ax.set_xlim(-10000)
    ax.set_xticks([0, 250000, 500000, 750000])
    ax.legend()
    sns.despine(fig, ax)
    fig.savefig(save_folder+"test_errors_th.pdf", type="pdf")

    std_zf = c.GradientData.load_standards("gd_training_data.hdf5")
    ana_th = a.Analyzer(MoTypes(False), std_zf, "sim_store_tanh.hdf5", "activity_store_tanh.hdf5")
    ana_zf = a.Analyzer(MoTypes(False), std_zf, "sim_store.hdf5", "activity_store.hdf5")

    # second panel: Gradient distribution
    bns = np.linspace(0, GlobalDefs.circle_sim_params["radius"], 100)
    centers = a.temp_convert(bns[:-1]+np.diff(bns), "r")
    naive = np.empty((len(paths_512_th), centers.size))
    trained_th = np.empty_like(naive)
    trained_zf = np.empty((len(paths_512_zf), centers.size))
    for i, p in enumerate(paths_512_th):
        pos_n = ana_th.run_simulation(mpath(base_path_th, p), "r", "naive")
        naive[i, :] = a.bin_simulation(pos_n, bns, "r")
        pos_t = ana_th.run_simulation(mpath(base_path_th, p), "r", "trained")
        trained_th[i, :] = a.bin_simulation(pos_t, bns, "r")
    for i, p in enumerate(paths_512_zf):
        pos_t = ana_zf.run_simulation(mpath(base_path_zf, p), "r", "trained")
        trained_zf[i, :] = a.bin_simulation(pos_t, bns, "r")
    fig, ax = pl.subplots()
    sns.tsplot(naive, centers, n_boot=1000, condition="Naive", color='k')
    sns.tsplot(trained_th, centers, n_boot=1000, condition="Trained", color="C1")
    ax.plot(centers, np.mean(trained_zf, 0), 'k', lw=0.25)
    ax.plot([GlobalDefs.tPreferred, GlobalDefs.tPreferred], [0, 0.03], 'k--', lw=0.25)
    ax.legend()
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel("Proportion")
    sns.despine(fig, ax)
    fig.savefig(save_folder+"gradient_distribution_th.pdf", type="pdf")

    # load and interpolate temperature stimulus
    dfile = h5py.File("stimFile.hdf5", 'r')
    tsin = np.array(dfile['sine_L_H_temp'])
    x = np.arange(tsin.size)  # stored at 20 Hz !
    xinterp = np.linspace(0, tsin.size, tsin.size * GlobalDefs.frame_rate // 20)
    temperature = np.interp(xinterp, x, tsin)
    dfile.close()

    # get activity data
    all_ids_th = []
    all_cells_th = []
    for i, p in enumerate(paths_512_th):
        cell_res, ids = ana_th.temperature_activity(mpath(base_path_th, p), temperature, i)
        all_ids_th.append(ids)
        all_cells_th.append(cell_res)
    all_ids_th = np.hstack(all_ids_th)
    all_cells_th = np.hstack(all_cells_th)
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
    for i in range(all_cells_th.shape[1]):
        all_cells_th[:, i] = convolve(all_cells_th[:, i], kernel, mode='full')[:all_cells_th.shape[0]]
    for i in range(all_cells_zf.shape[1]):
        all_cells_zf[:, i] = convolve(all_cells_zf[:, i], kernel, mode='full')[:all_cells_zf.shape[0]]

    # load cluster data from file
    clust_ids_th = a.cluster_activity(8, all_cells_th, "cluster_info_tanh.hdf5")[0]
    clust_ids_zf = a.cluster_activity(8, all_cells_zf, "cluster_info.hdf5")[0]

    # plot colors
    pal = sns.color_palette()  # the default matplotlib color cycle
    plot_cols_th = {0: pal[0], 1: pal[1], 2: pal[2], 3: pal[3], 4: pal[4], 5: pal[5],
                    6: pal[6], 7: pal[7], -1: (0.6, 0.6, 0.6)}

    # panel - all cluster activities, sorted into ON and anti-correlated OFF types
    n_regs_th = np.unique(clust_ids_th).size - 1
    n_regs_zf = np.unique(clust_ids_zf).size - 1
    cluster_acts_th = np.zeros((all_cells_th.shape[0] // 3, n_regs_th))
    is_on = np.zeros(n_regs_th, dtype=bool)
    ax_ix = np.full(n_regs_th, -1, dtype=int)
    on_count = 0
    for i in range(n_regs_th):
        act = np.mean(a.trial_average(all_cells_th[:, clust_ids_th == i], 3), 1)
        cluster_acts_th[:, i] = act
        is_on[i] = np.corrcoef(act, temperature[:act.size])[0, 1] > 0
        # correspondin axis on ON plot is simply set by order of cluster occurence
        if is_on[i]:
            ax_ix[i] = 0 if on_count < 2 else 1
            on_count += 1
    # for off types, put them on the corresponding off axis of the most anti-correlated ON type
    type_corrs_th = np.corrcoef(cluster_acts_th.T)
    for i in range(n_regs_th):
        if not is_on[i]:
            corresponding_on = np.argmin(type_corrs_th[i, :])
            assert is_on[corresponding_on]
            ax_ix[i] = ax_ix[corresponding_on]
    fig, (axes_on, axes_off) = pl.subplots(ncols=2, nrows=2, sharey=True, sharex=True)
    time = np.arange(cluster_acts_th.shape[0]) / GlobalDefs.frame_rate
    for i in range(n_regs_th):
        act = cluster_acts_th[:, i]
        if not is_on[i]:
            ax_off = axes_off[ax_ix[i]]
            ax_off.plot(time, cluster_acts_th[:, i], color=plot_cols_th[i])
        else:
            ax_on = axes_on[ax_ix[i]]
            ax_on.plot(time, cluster_acts_th[:, i], color=plot_cols_th[i])
    axes_off[0].set_xticks([0, 30, 60, 90, 120, 150])
    axes_off[1].set_xticks([0, 30, 60, 90, 120, 150])
    axes_off[0].set_xlabel("Time [s]")
    axes_off[1].set_xlabel("Time [s]")
    axes_on[0].set_ylabel("Cluster average activation")
    axes_off[0].set_ylabel("Cluster average activation")
    sns.despine()
    fig.tight_layout()
    fig.savefig(save_folder + "th_all_cluster_averages.pdf", type="pdf")

    # panel - average type counts in temperature branch for each cluster
    cl_type_d = {"Fraction": [], "net_id": [], "Cluster ID": [], "Layer": []}
    for i in range(len(paths_512_th)):
        for j in range(-1, n_regs_th):
            for k in range(2):
                lay_clust_ids = clust_ids_th[np.logical_and(all_ids_th[0, :] == i, all_ids_th[1, :] == k)]
                cl_type_d["Fraction"].append(np.sum(lay_clust_ids == j) / 512)
                cl_type_d["net_id"].append(i)
                cl_type_d["Cluster ID"].append(j)
                cl_type_d["Layer"].append(k)
    cl_type_df = DataFrame(cl_type_d)
    fig, (ax_0, ax_1) = pl.subplots(nrows=2, sharex=True)
    sns.barplot("Cluster ID", "Fraction", data=cl_type_df[cl_type_df["Layer"] == 0],
                order=list(range(n_regs_th)) + [-1], ci=68, ax=ax_0, palette=plot_cols_th)
    sns.barplot("Cluster ID", "Fraction", data=cl_type_df[cl_type_df["Layer"] == 1],
                order=list(range(n_regs_th)) + [-1], ci=68, ax=ax_1, palette=plot_cols_th)
    ax_0.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax_1.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    sns.despine(fig)
    fig.savefig(save_folder + "th_all_cluster_counts.pdf", type="pdf")

    # panel - cumulative explained variance by ReLu and Tanh PCs
    cluster_acts_zf = np.zeros((all_cells_zf.shape[0] // 3, n_regs_zf))
    for i in range(n_regs_zf):
        act = np.mean(a.trial_average(all_cells_zf[:, clust_ids_zf == i], 3), 1)
        cluster_acts_zf[:, i] = act
    type_corrs_zf = np.corrcoef(cluster_acts_zf.T)

    pca_zf = PCA(n_components=20)
    pca_zf.fit(a.trial_average(all_cells_zf, 3).T)
    pca_th = PCA(n_components=20)
    pca_th.fit(a.trial_average(all_cells_th, 3).T)

    fig, ax = pl.subplots()
    ax.plot(np.arange(20) + 1, np.cumsum(pca_zf.explained_variance_ratio_)*100, '.', label='ReLu')
    ax.plot(np.arange(20) + 1, np.cumsum(pca_th.explained_variance_ratio_)*100, '.', label='tanh')
    ax.plot([1, 20], [100, 100], 'k--', lw=0.25)
    ax.plot([1, 20], [99, 99], 'k--', lw=0.25)
    ax.legend()
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Cumulative explained variance [%]")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "th_zf_pca_cumvar_comp.pdf", type="pdf")
