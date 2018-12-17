#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script for panels of Figure 1 (Zebrafish model training, evolution and navigation)
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


# file definitions
base_path = "./model_data/Adam_1e-4/sepInput_mixTrain/"

paths_1024 = [f+'/' for f in os.listdir(base_path) if "_3m1024_" in f]
paths_512 = [f+'/' for f in os.listdir(base_path) if "_3m512_" in f]
paths_256 = [f+'/' for f in os.listdir(base_path) if "_3m256_" in f]


def test_loss(path):
    fname = base_path + path + "losses.hdf5"
    lossfile = h5py.File(fname, "r")
    test_losses = np.array(lossfile["test_losses"])
    rank_errors = np.array(lossfile["test_rank_errors"])
    timepoints = np.array(lossfile["test_eval"])
    return timepoints, test_losses, rank_errors


def ev_path(path):
    return base_path + path + "evolve/"


def mpath(path):
    return base_path + path[:-1]  # need to remove trailing slash


def compute_gradient_bout_frequency(model_path, drop_list=None):
    def bout_freq(pos: np.ndarray):
        r = np.sqrt(np.sum(pos[:, :2]**2, 1))  # radial position
        spd = np.r_[0, np.sqrt(np.sum(np.diff(pos[:, :2], axis=0) ** 2, 1))]  # speed
        bs = np.r_[0, np.diff(spd) > 0.00098]  # bout starts
        bins = np.linspace(0, GlobalDefs.circle_sim_params["radius"], 6)
        bcenters = bins[:-1] + np.diff(bins)/2
        cnt_r = np.histogram(r, bins)[0]
        cnt_r_bs = np.histogram(r[bs > 0.1], bins)[0]
        bfreq = cnt_r_bs / cnt_r * GlobalDefs.frame_rate
        return bfreq, bcenters

    with SimulationStore("sim_store.hdf5", std, MoTypes(False)) as sim_store:
        pos_fixed = sim_store.get_sim_pos(model_path, "r", "trained", drop_list)
        pos_part = sim_store.get_sim_pos(model_path, "r", "partevolve", drop_list)
        pos_var = sim_store.get_sim_pos(model_path, "r", "bfevolve", drop_list)
    bf_fixed, bc = bout_freq(pos_fixed)
    bf_p, bc = bout_freq(pos_part)
    bf_var, bc = bout_freq(pos_var)
    return bc, bf_fixed, bf_p, bf_var


if __name__ == "__main__":
    save_folder = "./DataFigures/Figure1/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42

    std = c.GradientData.load_standards("gd_training_data.hdf5")

    # first panel - log squared error progression over training
    test_time = test_loss(paths_512[0])[0]
    test_256 = np.vstack([test_loss(lp)[1] for lp in paths_256])
    test_512 = np.vstack([test_loss(lp)[1] for lp in paths_512])
    test_1024 = np.vstack([test_loss(lp)[1] for lp in paths_1024])
    fig, ax = pl.subplots()
    sns.tsplot(np.log10(test_256), test_time, ax=ax, color="C2", n_boot=1000, condition="256 HU")
    sns.tsplot(np.log10(test_512), test_time, ax=ax, color="C1", n_boot=1000, condition="512 HU")
    sns.tsplot(np.log10(test_1024), test_time, ax=ax, color="C3", n_boot=1000, condition="1024 HU")
    epoch_times = np.linspace(0, test_time.max(), 10, endpoint=False)
    for e in epoch_times:
        ax.plot([e, e], [-1.2, .4], 'k--', lw=0.25)
    ax.set_ylabel("log(Squared test error)")
    ax.set_xlabel("Training step")
    ax.set_xlim(-10000)
    ax.set_xticks([0, 250000, 500000, 750000])
    ax.legend()
    sns.despine(fig, ax)
    fig.savefig(save_folder+"test_errors.pdf", type="pdf")

    print("Prediction error: {0} C +/- {1} C".format(np.mean(test_512[:, -10:]), np.std(test_512[:, -10:])))

    # second panel - population average temperature error progression during evolution
    errors = np.empty((len(paths_512), 50))
    for i, p in enumerate(paths_512):
        errors[i, :] = np.mean(np.load(ev_path(p)+"generation_errors.npy"), 1)
    fig, ax = pl.subplots()
    sns.tsplot(errors, np.arange(50), n_boot=1000, color="C1", err_style="ci_bars", interpolate=False)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Navigation error [C]")
    ax.set_xlim(-1, 50)
    ax.set_xticks([0, 25, 50])
    sns.despine(fig, ax)
    fig.savefig(save_folder+"evolution_nav_errors.pdf", type="pdf")

    # third panel - bout frequency modulation with and without evolution
    bf_trained = np.empty((len(paths_512), 5))
    bf_part = np.empty_like(bf_trained)
    bf_evolved = np.empty_like(bf_trained)
    centers = None
    for i, p in enumerate(paths_512):
        centers, t, part, e = compute_gradient_bout_frequency(mpath(p))
        bf_trained[i, :] = t
        bf_part[i, :] = part
        bf_evolved[i, :] = e
    centers = a.temp_convert(centers, "r")
    fig, ax = pl.subplots()
    sns.tsplot(bf_trained, centers, n_boot=1000, color="C1", err_style="ci_band", condition="Generation 0")
    sns.tsplot(bf_part, centers, n_boot=1000, color=(.5, .5, .5), err_style="ci_band", condition="Generation 8")
    sns.tsplot(bf_evolved, centers, n_boot=1000, color="C0", err_style="ci_band", condition="Generation 50")
    ax.set_xlim(23, 36)
    ax.set_xticks([25, 30, 35])
    ax.set_yticks([0.5, 0.75, 1, 1.25])
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel("Swim frequency [Hz]")
    ax.legend()
    sns.despine(fig, ax)
    fig.savefig(save_folder + "gradient_swim_frequency.pdf", type="pdf")

    # fourth panel - gradient distribution naive, trained, evolved
    bns = np.linspace(0, GlobalDefs.circle_sim_params["radius"], 100)
    centers = a.temp_convert(bns[:-1]+np.diff(bns), "r")
    ana = a.Analyzer(MoTypes(False), std, "sim_store.hdf5", None)
    naive = np.empty((len(paths_512), centers.size))
    trained = np.empty_like(naive)
    evolved = np.empty_like(naive)
    naive_errors = []
    trained_errors = []
    evolved_errors = []
    for i, p in enumerate(paths_512):
        pos_n = ana.run_simulation(mpath(p), "r", "naive")
        naive_errors.append(a.temp_error(pos_n, 'r'))
        naive[i, :] = a.bin_simulation(pos_n, bns, "r")
        pos_t = ana.run_simulation(mpath(p), "r", "trained")
        trained_errors.append(a.temp_error(pos_t, 'r'))
        trained[i, :] = a.bin_simulation(pos_t, bns, "r")
        pos_e = ana.run_simulation(mpath(p), "r", "bfevolve")
        evolved_errors.append(a.temp_error(pos_e, 'r'))
        evolved[i, :] = a.bin_simulation(pos_e, bns, "r")
    print("Naive erorr = {0} C +/- {1} C".format(np.mean(naive_errors), np.std(naive_errors)))
    print("Trained erorr = {0} C +/- {1} C".format(np.mean(trained_errors), np.std(trained_errors)))
    print("Evolved erorr = {0} C +/- {1} C".format(np.mean(evolved_errors), np.std(evolved_errors)))
    fig, ax = pl.subplots()
    sns.tsplot(naive, centers, n_boot=1000, condition="Naive", color='k')
    sns.tsplot(trained, centers, n_boot=1000, condition="Trained", color="C1")
    sns.tsplot(evolved, centers, n_boot=1000, condition="p(Swim) control", color="C0")
    ax.plot([GlobalDefs.tPreferred, GlobalDefs.tPreferred], [0, 0.05], 'k--', lw=0.25)
    ax.legend()
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel("Proportion")
    sns.despine(fig, ax)
    fig.savefig(save_folder+"gradient_distribution.pdf", type="pdf")
