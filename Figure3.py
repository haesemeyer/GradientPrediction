#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script for panels of Figure 3 (C elegans model training, evolution and navigation and zebrafish unit comparison)
"""

import core as c
import analysis as a
from global_defs import GlobalDefs
import os
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as pl
import numpy as np
import h5py
from data_stores import SimulationStore
from mo_types import MoTypes
from Figure4 import mpath
from sklearn.decomposition import PCA
from scipy.signal import convolve


# file definitions
base_path_zf = "./model_data/Adam_1e-4/sepInput_mixTrain/"
paths_512_zf = [f + '/' for f in os.listdir(base_path_zf) if "_3m512_" in f]

base_path_ce = "./model_data/CE_Adam_1e-4/"
paths_512_ce = [f + '/' for f in os.listdir(base_path_ce) if "_3m512_" in f]


def test_loss(path):
    fname = base_path_ce + path + "losses.hdf5"
    lossfile = h5py.File(fname, "r")
    test_losses = np.array(lossfile["test_losses"])
    rank_errors = np.array(lossfile["test_rank_errors"])
    timepoints = np.array(lossfile["test_eval"])
    return timepoints, test_losses, rank_errors


def plot_pc(index):
    global coords
    f, axis = pl.subplots()
    sns.kdeplot(coords[species_id == 0, index]/np.sqrt(pca.explained_variance_[i]), shade=True, ax=axis)
    sns.kdeplot(coords[species_id == 1, index]/np.sqrt(pca.explained_variance_[i]), shade=True, ax=axis)
    axis.set_xlabel("PC {0}".format(index+1))
    axis.set_ylabel("Density")
    sns.despine(f, axis)
    f.savefig(save_folder + "PC_SpaceComparison_PC{0}.pdf".format(index+1), type="pdf")


if __name__ == "__main__":
    save_folder = "./DataFigures/Figure3/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42

    std_zf = c.GradientData.load_standards("gd_training_data.hdf5")
    ana_zf = a.Analyzer(MoTypes(False), std_zf, "sim_store.hdf5", "activity_store.hdf5")
    std_ce = c.GradientData.load_standards("ce_gd_training_data.hdf5")
    ana_ce = a.Analyzer(MoTypes(True), std_ce, "ce_sim_store.hdf5", "ce_activity_store.hdf5")

    # load activity clusters from file
    clfile = h5py.File("cluster_info.hdf5", "r")
    clust_ids_zf = np.array(clfile["clust_ids"])
    clfile.close()
    clfile = h5py.File("ce_cluster_info.hdf5", "r")
    clust_ids_ce = np.array(clfile["clust_ids"])
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
    all_ids_ce = []
    all_cells_ce = []
    for i, p in enumerate(paths_512_ce):
        cell_res, ids = ana_ce.temperature_activity(mpath(base_path_ce, p), temperature, i)
        all_ids_ce.append(ids)
        all_cells_ce.append(cell_res)
    all_ids_ce = np.hstack(all_ids_ce)
    all_cells_ce = np.hstack(all_cells_ce)

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
    for i in range(all_cells_ce.shape[1]):
        all_cells_ce[:, i] = convolve(all_cells_ce[:, i], kernel, method='full')[:all_cells_ce.shape[0]]

    # first panel - rank error progression over training
    test_time = test_loss(paths_512_ce[0])[0]
    test_512 = np.vstack([test_loss(lp)[2] for lp in paths_512_ce])
    fig, ax = pl.subplots()
    sns.tsplot(test_512, test_time, ax=ax, color="C1", n_boot=1000, condition="512 HU")
    epoch_times = np.linspace(0, test_time.max(), 10, endpoint=False)
    for e in epoch_times:
        ax.plot([e, e], [0, 8], 'k--', lw=0.25)
    ax.plot([0, test_time.max()], [8, 8], 'k--', lw=0.25)
    ax.set_ylabel("Ranking error")
    ax.set_xlabel("Training step")
    ax.set_xlim(-10000)
    ax.set_xticks([0, 250000, 500000])
    ax.legend()
    sns.despine(fig, ax)
    fig.savefig(save_folder+"test_rank_errors.pdf", type="pdf")

    # second panel - gradient distribution naive, trained
    bns = np.linspace(0, GlobalDefs.circle_sim_params["radius"], 100)
    centers = a.temp_convert(bns[:-1]+np.diff(bns), "r")
    naive = np.empty((len(paths_512_ce), centers.size))
    trained = np.empty_like(naive)
    for i, p in enumerate(paths_512_ce):
        pos_n = ana_ce.run_simulation(mpath(base_path_ce, p), "r", "naive")
        naive[i, :] = a.bin_simulation(pos_n, bns, "r")
        pos_t = ana_ce.run_simulation(mpath(base_path_ce, p), "r", "trained")
        trained[i, :] = a.bin_simulation(pos_t, bns, "r")
    fig, ax = pl.subplots()
    sns.tsplot(naive, centers, n_boot=1000, condition="Naive", color='k')
    sns.tsplot(trained, centers, n_boot=1000, condition="Trained", color="C1")
    ax.plot([GlobalDefs.tPreferred, GlobalDefs.tPreferred], [0, 0.075], 'k--', lw=0.25)
    ax.legend()
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel("Proportion")
    ax.set_yticks([0, 0.025, 0.05, 0.075])
    sns.despine(fig, ax)
    fig.savefig(save_folder+"gradient_distribution.pdf", type="pdf")

    # third panel - responses of C elegans neurons to temperature step
    afd_like = 1
    awc_like = 7
    step_min = 23
    step_max = 27
    temp_step = np.zeros(temperature.size // 3)
    temp_step[:temp_step.size//5] = step_min
    temp_step[temp_step.size*4//5:] = step_max
    ramp = temp_step[temp_step.size//5:temp_step.size*4//5]
    ramp = np.arange(ramp.size)/ramp.size*(step_max-step_min) + step_min
    temp_step[temp_step.size//5:temp_step.size*4//5] = ramp
    cells_ce_step = []
    for i, p in enumerate(paths_512_ce):
        cell_res, ids = ana_ce.temperature_activity(mpath(base_path_ce, p), temp_step, i)
        cells_ce_step.append(cell_res)
    cells_ce_step = np.hstack(cells_ce_step)
    for i in range(cells_ce_step.shape[1]):
        cells_ce_step[:, i] = convolve(cells_ce_step[:, i], kernel, method='full')[:cells_ce_step.shape[0]]
    afd_data = cells_ce_step[:, clust_ids_ce == afd_like].T
    awc_data = cells_ce_step[:, clust_ids_ce == awc_like].T
    trial_time = np.arange(cells_ce_step.shape[0]) / GlobalDefs.frame_rate
    fig, ax = pl.subplots()
    sns.tsplot(afd_data, trial_time, ax=ax, color="C3")
    sns.tsplot(awc_data, trial_time, ax=ax, color="C1")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Activation [AU]")
    ax.set_xticks([0, 30, 60, 90, 120, 150])
    sns.despine(fig, ax)
    fig.savefig(save_folder + "ce_step_responses.pdf", type="pdf")
    fig, ax = pl.subplots()
    ax.plot(trial_time, temp_step, 'k')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [C]")
    ax.set_xticks([0, 30, 60, 90, 120, 150])
    sns.despine(fig, ax)
    fig.savefig(save_folder + "step_stimulus.pdf", type="pdf")


    # fourth panel - comparison of cell response location in PCA space
    all_cells = np.hstack((a.trial_average(all_cells_zf, 3), a.trial_average(all_cells_ce, 3))).T
    max_vals = np.max(all_cells, 1, keepdims=True)
    max_vals[max_vals == 0] = 1  # these cells do not show any response
    all_cells /= max_vals
    species_id = np.zeros(all_cells.shape[0])
    species_id[all_cells_zf.shape[1]:] = 1
    pca = PCA(4)
    pca.fit(all_cells)
    coords = pca.transform(all_cells)
    for i in range(pca.n_components):
        plot_pc(i)
