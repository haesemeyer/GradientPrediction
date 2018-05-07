#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script for panels of Figure S3 (C elegans model training, evolution and navigation and zebrafish unit comparison)
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
from scipy.signal import convolve
from sklearn.decomposition import PCA


# file definitions
base_path_zf = "./model_data/Adam_1e-4/sepInput_mixTrain/"
paths_512_zf = [f + '/' for f in os.listdir(base_path_zf) if "_3m512_" in f]

base_path_ce = "./model_data/CE_Adam_1e-4/"
paths_512_ce = [f + '/' for f in os.listdir(base_path_ce) if "_3m512_" in f]

base_path_pt = "./model_data/Phototaxis/"
paths_512_pt = [f + '/' for f in os.listdir(base_path_pt) if "_3m512_" in f]


def plot_pc(index):
    global coords
    global species_id
    global pca
    f, axis = pl.subplots()
    sns.kdeplot(coords[species_id == 0, index]/np.sqrt(pca.explained_variance_[i]), shade=True, ax=axis)
    sns.kdeplot(coords[species_id == 1, index]/np.sqrt(pca.explained_variance_[i]), shade=True, ax=axis)
    axis.set_xlabel("PC {0}".format(index+1))
    axis.set_ylabel("Density")
    sns.despine(f, axis)
    f.savefig(save_folder + "PC_SpaceComparison_PC{0}.pdf".format(index+1), type="pdf")


# The following functions are necessary since phototaxis simulations are currently not part of the data store scheme
def do_simulation(path):
    """
    Uses a model identified by path to run a naive and a trained simulation
    :param path: The model path
    :return:
        [0]: The facing angle bin centers
        [1]: The occupancy of the naive model
        [2]: The occupancy of the trained model
    """
    global std_pt
    bins = np.linspace(-np.pi, np.pi, 100)
    # bin-centers in degress
    bcenters = bins[:-1]+np.diff(bins)/2
    # naive simulation
    mdata = c.ModelData(path)
    model_naive = c.ZfGpNetworkModel()
    model_naive.load(mdata.ModelDefinition, mdata.FirstCheckpoint)
    model_trained = c.ZfGpNetworkModel()
    model_trained.load(mdata.ModelDefinition, mdata.LastCheckpoint)
    sim = MoTypes(False).pt_sim(model_naive, std_pt, 100)
    pos_naive = sim.run_simulation(GlobalDefs.n_steps)
    h_naive = a.bin_simulation_pt(pos_naive, bins)
    sim = MoTypes(False).pt_sim(model_trained, std_pt, 100)
    pos_trained = sim.run_simulation(GlobalDefs.n_steps)
    h_trained = a.bin_simulation_pt(pos_trained, bins)
    return bcenters, h_naive, h_trained


def get_cell_responses(path, temp):
    """
    Loads a model and computes the temperature response of all neurons returning response matrix
    :param path: Model path
    :param temp: Temperature stimulus
    :return: n-timepoints x m-neurons matrix of responses
    """
    global std_pt
    mdata = c.ModelData(path)
    # create our model and load from last checkpoint
    gpn = c.ZfGpNetworkModel()
    gpn.load(mdata.ModelDefinition, mdata.LastCheckpoint)
    # prepend lead-in to stimulus
    lead_in = np.full(gpn.input_dims[2] - 1, np.mean(temp[:10]))
    temp = np.r_[lead_in, temp]
    activities = gpn.unit_stimulus_responses(temp, None, None, std_pt)
    return np.hstack(activities['t']) if 't' in activities else np.hstack(activities['m'])


if __name__ == "__main__":
    save_folder = "./DataFigures/FigureS3/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42

    std_zf = c.GradientData.load_standards("gd_training_data.hdf5")
    ana_zf = a.Analyzer(MoTypes(False), std_zf, "sim_store.hdf5", "activity_store.hdf5")
    std_ce = c.GradientData.load_standards("ce_gd_training_data.hdf5")
    ana_ce = a.Analyzer(MoTypes(True), std_ce, "ce_sim_store.hdf5", "ce_activity_store.hdf5")

    std_pt = c.GradientData.load_standards("photo_training_data.hdf5")

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
    trial_time = np.arange(all_cells_zf.shape[0] // 3) / GlobalDefs.frame_rate

    # Panels: Plot best-correlated pairs of zebrafish and C. elegans neurons
    clust_corr_mat = np.zeros((np.unique(clust_ids_zf).size-1, np.unique(clust_ids_ce).size-1))
    regs_ce = {}
    for i in range(clust_corr_mat.shape[0]):
        reg_zf = np.mean(all_cells_zf[:, clust_ids_zf == i], 1)
        for j in range(clust_corr_mat.shape[1]):
            if j in regs_ce:
                clust_corr_mat[i, j] = np.corrcoef(reg_zf, regs_ce[j])[0, 1]**2
            else:
                r = np.mean(all_cells_ce[:, clust_ids_ce == j], 1)
                regs_ce[j] = r
                clust_corr_mat[i, j] = np.corrcoef(reg_zf, r)[0, 1]**2

    def norm(trace):
        n = trace - trace.min()
        return n / n.max()

    ccm_copy = clust_corr_mat.copy()
    for i in range(np.max(ccm_copy.shape)):
        ix = np.argmax(ccm_copy)
        rw, cl = np.unravel_index(ix, ccm_copy.shape)
        ccm_copy[rw, :] = 0
        ccm_copy[:, cl] = 0
        fig, ax = pl.subplots()
        ax.plot(trial_time, norm(np.mean(a.trial_average(all_cells_zf[:, clust_ids_zf == rw], 3), 1)), color='k')
        ax.plot(trial_time, norm(np.mean(a.trial_average(all_cells_ce[:, clust_ids_ce == cl], 3), 1)), color='C1')
        ax.set_ylabel("Normalized activation")
        ax.set_xlabel("Time [s]")
        ax.set_title("Zf {0} vs Ce {1}. R^2 = {2}".format(rw, cl, np.round(clust_corr_mat[rw, cl], 2)))
        ax.set_xticks([0, 30, 60, 90, 120, 150])
        sns.despine(fig, ax)
        fig.savefig(save_folder + "ZFish_C{0}_vs_CElegans_C{1}.pdf".format(rw, cl), type="pdf")

    # Panel - naive and trained phototaxis performance
    all_n = []
    t_512 = []
    bins = None
    for p512 in paths_512_pt:
        bins, naive, trained = do_simulation(mpath(base_path_pt, p512))[:3]
        all_n.append(naive)
        t_512.append(trained)
    t_512 = np.vstack(t_512)
    all_n = np.vstack(all_n)
    fig, ax = pl.subplots()
    sns.tsplot(t_512, bins, n_boot=1000, ax=ax, color="C1")
    ax.plot(bins, np.mean(t_512, 0), lw=2, label="512 HU", c="C1")
    sns.tsplot(all_n, bins, n_boot=1000, ax=ax, color="k")
    ax.plot(bins, np.mean(all_n, 0), "k", lw=2, label="Naive")
    ax.plot([0, 0], ax.get_ylim(), 'C4--')
    ax.set_ylim(0)
    ax.legend()
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Angle to light source")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "Phototaxis_navigation.pdf", type="pdf")

    # Panel - PC space comparison of heat zebrafish and phototaxis zebrafish cells
    all_cells_pt = []
    for p in paths_512_pt:
        all_cells_pt.append(get_cell_responses(mpath(base_path_pt, p), temperature))
    all_cells_pt = np.hstack(all_cells_pt)
    for i in range(all_cells_pt.shape[1]):
        all_cells_pt[:, i] = convolve(all_cells_pt[:, i], kernel, method='full')[:all_cells_pt.shape[0]]

    all_cells = np.hstack((a.trial_average(all_cells_zf, 3), a.trial_average(all_cells_pt, 3))).T
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
