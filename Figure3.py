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

    # second panel - gradient distribution naive, trained, evolved
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
