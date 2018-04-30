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


# file definitions
base_path_zf = "./model_data/Adam_1e-4/sepInput_mixTrain/"
paths_512_zf = [f + '/' for f in os.listdir(base_path_zf) if "_3m512_" in f]

base_path_ce = "./model_data/CE_Adam_1e-4/"
paths_512_ce = [f + '/' for f in os.listdir(base_path_ce) if "_3m512_" in f]


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
        r, c = np.unravel_index(ix, ccm_copy.shape)
        ccm_copy[r, :] = 0
        ccm_copy[:, c] = 0
        fig, ax = pl.subplots()
        ax.plot(trial_time, norm(np.mean(a.trial_average(all_cells_zf[:, clust_ids_zf == r], 3), 1)), color='k')
        ax.plot(trial_time, norm(np.mean(a.trial_average(all_cells_ce[:, clust_ids_ce == c], 3), 1)), color='C1')
        ax.set_ylabel("Normalized activation")
        ax.set_xlabel("Time [s]")
        ax.set_title("Zf {0} vs Ce {1}. R^2 = {2}".format(r, c, np.round(clust_corr_mat[r, c], 2)))
        ax.set_xticks([0, 30, 60, 90, 120, 150])
        sns.despine(fig, ax)
        fig.savefig(save_folder + "ZFish_C{0}_vs_CElegans_C{1}.pdf".format(r, c), type="pdf")
