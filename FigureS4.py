#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script for panels of Figure S4 (Ablations in C elegans and zebrafish networks)
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


# file definitions
base_path_zf = "./model_data/Adam_1e-4/sepInput_mixTrain/"
paths_512_zf = [f + '/' for f in os.listdir(base_path_zf) if "_3m512_" in f]

base_path_ce = "./model_data/CE_Adam_1e-4/"
paths_512_ce = [f + '/' for f in os.listdir(base_path_ce) if "_3m512_" in f]


def test_loss_zf_retrain(path):
    fname = base_path_zf + path + "fl_nontbranch_retrain/losses.hdf5"
    lossfile = h5py.File(fname, "r")
    rank_errors_non_t = np.array(lossfile["test_rank_errors"])
    fname = base_path_zf + path + "fl_tbranch_retrain/losses.hdf5"
    lossfile = h5py.File(fname, "r")
    rank_errors_t = np.array(lossfile["test_rank_errors"])
    timepoints = np.array(lossfile["test_eval"])
    return timepoints, rank_errors_t, rank_errors_non_t


def test_loss_ce_retrain(path):
    fname = base_path_ce + path + "cel_nontbranch_retrain/losses.hdf5"
    lossfile = h5py.File(fname, "r")
    rank_errors_non_t = np.array(lossfile["test_rank_errors"])
    fname = base_path_ce + path + "cel_tbranch_retrain/losses.hdf5"
    lossfile = h5py.File(fname, "r")
    rank_errors_t = np.array(lossfile["test_rank_errors"])
    timepoints = np.array(lossfile["test_eval"])
    return timepoints, rank_errors_t, rank_errors_non_t


if __name__ == "__main__":
    save_folder = "./DataFigures/FigureS4/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42

    std_zf = c.GradientData.load_standards("gd_training_data.hdf5")

    std_ce = c.GradientData.load_standards("ce_gd_training_data.hdf5")
    ana_ce = a.Analyzer(MoTypes(True), std_ce, "ce_sim_store.hdf5", "ce_activity_store.hdf5")
    ana_zf = a.Analyzer(MoTypes(False), std_zf, "sim_store.hdf5", "activity_store.hdf5")

    # load cluster data from file
    clfile = h5py.File("ce_cluster_info.hdf5", "r")
    clust_ids_ce = np.array(clfile["clust_ids"])
    clfile.close()
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
    all_ids_ce = []
    all_cells_ce = []
    for i, p in enumerate(paths_512_ce):
        cell_res, ids = ana_ce.temperature_activity(mpath(base_path_ce, p), temperature, i)
        all_ids_ce.append(ids)
        all_cells_ce.append(cell_res)
    all_ids_ce = np.hstack(all_ids_ce)
    all_cells_ce = np.hstack(all_cells_ce)

    all_ids_zf = []
    for i, p in enumerate(paths_512_zf):
        cell_res, ids = ana_zf.temperature_activity(mpath(base_path_zf, p), temperature, i)
        all_ids_zf.append(ids)
    all_ids_zf = np.hstack(all_ids_zf)

    # panel X: Zebrafish Rank error progress during re-training
    re_t_branch = []
    re_nont_branch = []
    test_times = None
    for p in paths_512_zf:
        test_times, t, non_t = test_loss_zf_retrain(p)
        re_t_branch.append(t)
        re_nont_branch.append(non_t)
    re_t_branch = np.vstack(re_t_branch)
    re_nont_branch = np.vstack(re_nont_branch)
    fig, ax = pl.subplots()
    sns.tsplot(re_t_branch, test_times, ci=68, color="C1", condition="Temperature branch only")
    sns.tsplot(re_nont_branch, test_times, ci=68, color="C3", condition="Mixed branch only")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "zf_fish_retrain_rank_errors.pdf", type="pdf")

    # panel X: C elegans Rank error progress during re-training
    re_t_branch = []
    re_nont_branch = []
    test_times = None
    for p in paths_512_ce:
        test_times, t, non_t = test_loss_ce_retrain(p)
        re_t_branch.append(t)
        re_nont_branch.append(non_t)
    re_t_branch = np.vstack(re_t_branch)
    re_nont_branch = np.vstack(re_nont_branch)
    fig, ax = pl.subplots()
    sns.tsplot(re_t_branch, test_times, ci=68, color="C1", condition="Temperature branch only")
    sns.tsplot(re_nont_branch, test_times, ci=68, color="C3", condition="Mixed branch only")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "ce_fish_retrain_rank_errors.pdf", type="pdf")

    # panel X - white noise analysis after fish-like ablations to show that non-fish types modulate behavior
    mo = MoTypes(False)
    behav_kernels = {}
    k_names = ["stay", "straight", "left", "right"]
    for i, p in enumerate(paths_512_zf):
        m_path = mpath(base_path_zf, p)
        mdata_wn = c.ModelData(m_path)
        gpn_wn = mo.network_model()
        gpn_wn.load(mdata_wn.ModelDefinition, mdata_wn.LastCheckpoint)
        wna = mo.wn_sim(std_zf, gpn_wn, stim_std=2)
        wna.switch_mean = 5
        wna.switch_std = 1
        wna.remove = a.create_det_drop_list(i, clust_ids_zf, all_ids_zf, [1, 2, 3, 4, 5])
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
    fig.savefig(save_folder+"nonfish_only_white_noise_kernels.pdf", type="pdf")

    # for each removal and retrain of C elegans data compute gradient distributions
    bns = np.linspace(0, GlobalDefs.circle_sim_params["radius"], 100)
    centers = a.temp_convert(bns[:-1] + np.diff(bns), "r")
    trained = np.empty((len(paths_512_ce), centers.size))
    cel_ablated = np.empty_like(trained)
    cel_retrained_t = np.empty_like(trained)
    cel_retrained_nont = np.empty_like(trained)
    cenl_ablated = np.empty_like(trained)
    for i, p in enumerate(paths_512_ce):
        mp = mpath(base_path_ce, p)
        rt_path_nont = mp + "/cel_nontbranch_retrain"
        rt_path_t = mp + "/cel_tbranch_retrain"
        with SimulationStore(None, std_ce, MoTypes(True)) as sim_store:
            pos = sim_store.get_sim_pos(mp, 'r', "trained")
            trained[i, :] = a.bin_simulation(pos, bns, 'r')
            dlist = a.create_det_drop_list(i, clust_ids_ce, all_ids_ce, [1, 7])
            pos = sim_store.get_sim_pos(mp, 'r', "trained", dlist)
            cel_ablated[i, :] = a.bin_simulation(pos, bns, 'r')
            pos = sim_store.get_sim_pos(rt_path_t, 'r', "trained", dlist)
            cel_retrained_t[i, :] = a.bin_simulation(pos, bns, 'r')
            pos = sim_store.get_sim_pos(rt_path_nont, 'r', "trained", dlist)
            cel_retrained_nont[i, :] = a.bin_simulation(pos, bns, 'r')
            dlist = a.create_det_drop_list(i, clust_ids_ce, all_ids_ce, [0, 2, 3, 4, 5, 6])
            pos = sim_store.get_sim_pos(mp, 'r', "trained", dlist)
            cenl_ablated[i, :] = a.bin_simulation(pos, bns, 'r')

    # panel 6: Consequence of ablating all worm or non-worm types
    fig, ax = pl.subplots()
    sns.tsplot(trained, centers, n_boot=1000, condition="Trained", color="k")
    sns.tsplot(cel_ablated, centers, n_boot=1000, condition="Worm-like ablated", color="C1")
    sns.tsplot(cenl_ablated, centers, n_boot=1000, condition="Non-worm ablated", color="C3")
    ax.plot([GlobalDefs.tPreferred, GlobalDefs.tPreferred], [0, 0.075], 'k--', lw=0.25)
    ax.legend()
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel("Proportion")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "ce_fish_nonfish_ablation_distribution.pdf", type="pdf")

    # panel 7: Retraining after ablating all worm types
    fig, ax = pl.subplots()
    sns.tsplot(cel_ablated, centers, n_boot=1000, condition="Ablated", color="k")
    sns.tsplot(trained, centers, n_boot=1000, condition="Trained", color='k')
    sns.tsplot(cel_retrained_t, centers, n_boot=1000, condition="Temperature part retrained", color="C1")
    sns.tsplot(cel_retrained_nont, centers, n_boot=1000, condition="Shared part retrained", color="C3")
    ax.plot([GlobalDefs.tPreferred, GlobalDefs.tPreferred], [0, 0.075], 'k--', lw=0.25)
    ax.legend()
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel("Proportion")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "ce_fish_retrained_distribution.pdf", type="pdf")
