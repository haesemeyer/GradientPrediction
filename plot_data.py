#  Copyright 2017 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script to plot training progress and simulations and representations across previously
trained neural networks - this script is very data-set specific
"""

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as pl
import seaborn as sns
from gradientSimulation import run_simulation, CircleGradSimulation, LinearGradientSimulation
from core import ModelData, GradientData


# file definitions
base_path = "./model_data/Adam_1e-4/"

path_256 = "170829_NH_256_256_256/"
path_1024 = "170829_NH_1024_1024_1024/"
path_2048 = "170829_NH_2048_2048_2048/"
paths_512 = [
    "170829_NH_512_512_512/",
    "170901_512_512_512/",
    "170902_512_512_512/",
    "170903_512_512_512/",
    "170904_512_512_512/",
    "170905_512_512_512/",
    "170906_512_512_512/",
    "170907_512_512_512/"
]
# simulation globals
n_steps = 2000000
TPREFERRED = 25


def loss_file(path):
    return base_path + path + "losses.hdf5"


def mpath(path):
    return base_path + path[:-1]  # need to remove trailing slash


def train_loss(fname):
    dfile = h5py.File(fname, "r")
    train_losses = np.array(dfile["train_losses"])
    rank_errors = np.array(dfile["train_rank_errors"])
    timepoints = np.array(dfile["train_eval"])
    dfile.close()
    return timepoints, train_losses, rank_errors


def test_loss(fname):
    dfile = h5py.File(fname, "r")
    test_losses = np.array(dfile["test_losses"])
    rank_errors = np.array(dfile["test_rank_errors"])
    timepoints = np.arange(test_losses.size) * 1000
    return timepoints, test_losses, rank_errors


def plot_squared_losses():
    # assume timepoints same for all
    test_time, test_256 = test_loss(loss_file(path_256))[:2]
    test_512 = np.mean(np.vstack([test_loss(loss_file(p))[1] for p in paths_512]), 0)
    test_1024 = test_loss(loss_file(path_1024))[1]
    test_2048 = test_loss(loss_file(path_2048))[1]
    fig, ax = pl.subplots()
    ax.plot(test_time, np.log10(gaussian_filter1d(test_256, 10)), "C0.", label="256 HU")
    ax.plot(test_time, np.log10(gaussian_filter1d(test_512, 10)), "C1.", label="512 HU")
    ax.plot(test_time, np.log10(gaussian_filter1d(test_1024, 10)), "C2.", label="1024 HU")
    ax.plot(test_time, np.log10(gaussian_filter1d(test_2048, 10)), "C3.", label="2048 HU")
    epoch_times = np.linspace(0, test_time.max(), 10, endpoint=False)
    for e in epoch_times:
        ax.plot([e, e], [-1.5, -0.5], 'k--', lw=0.5)
    ax.set_ylabel("log(Squared test error)")
    ax.set_xlabel("Training step")
    ax.legend()
    sns.despine()


def plot_rank_losses():
    # assume timepoints same for all
    out = test_loss(loss_file(path_256))
    test_time, test_256 = out[0], out[2]
    test_512 = np.mean(np.vstack([test_loss(loss_file(p))[2] for p in paths_512]), 0)
    test_1024 = test_loss(loss_file(path_1024))[2]
    test_2048 = test_loss(loss_file(path_2048))[2]
    fig, ax = pl.subplots()
    ax.plot(test_time, gaussian_filter1d(test_256, 10), "C0.", label="256 HU")
    ax.plot(test_time, gaussian_filter1d(test_512, 10), "C1.", label="512 HU")
    ax.plot(test_time, gaussian_filter1d(test_1024, 10), "C2.", label="1024 HU")
    ax.plot(test_time, gaussian_filter1d(test_2048, 10), "C3.", label="2048 HU")
    epoch_times = np.linspace(0, test_time.max(), 10, endpoint=False)
    for e in epoch_times:
        ax.plot([e, e], [1.5, 4], 'k--', lw=0.5)
    ax.set_ylabel("Rank test error")
    ax.set_xlabel("Training step")
    ax.legend()
    sns.despine()


def do_simulation(path, train_data, sim_type, run_ideal):
    mdata = ModelData(path)
    if sim_type == "l":
        sim_type = "x"
        sim_naive = LinearGradientSimulation(mdata.ModelDefinition, mdata.FirstCheckpoint, train_data, 100, 100, 22, 37,
                                             TPREFERRED)
        sim_trained = LinearGradientSimulation(mdata.ModelDefinition, mdata.LastCheckpoint, train_data, 100, 100, 22,
                                               37, TPREFERRED)
    else:
        sim_naive = CircleGradSimulation(mdata.ModelDefinition, mdata.FirstCheckpoint, train_data, 100, 22, 37,
                                         TPREFERRED)
        sim_trained = CircleGradSimulation(mdata.ModelDefinition, mdata.LastCheckpoint, train_data, 100, 22, 37,
                                           TPREFERRED)
    b_naive, h_naive = run_simulation(sim_naive, n_steps, False, sim_type)[1:]
    b_trained, h_trained = run_simulation(sim_trained, n_steps, False, sim_type)[1:]
    if run_ideal:
        b_ideal, h_ideal = run_simulation(sim_trained, n_steps, True, sim_type)[1:]
        return b_naive, h_naive, h_trained, h_ideal
    else:
        return b_naive, h_naive, h_trained


def plot_sim(train_data, sim_type):
    all_n = []
    bins, n_256, t_256, ideal = do_simulation(mpath(path_256), train_data, sim_type, True)
    all_n.append(n_256)
    t_512 = []
    for p in paths_512:
        _, n, t = do_simulation(mpath(p), train_data, sim_type, False)
        all_n.append(n)
        t_512.append(t)
    t_512 = np.mean(np.vstack(t_512), 0)
    _, n_1024, t_1024 = do_simulation(mpath(path_1024), train_data, sim_type, False)
    all_n.append(n_1024)
    _, n_2048, t_2048 = do_simulation(mpath(path_2048), train_data, sim_type, False)
    all_n.append(n_2048)
    all_n = np.mean(np.vstack(all_n), 0)
    fig, ax = pl.subplots()
    ax.plot(bins, t_256, lw=2, label="256 HU")
    ax.plot(bins, t_512, lw=2, label="512 HU")
    ax.plot(bins, t_1024, lw=2, label="1024 HU")
    ax.plot(bins, t_2048, lw=2, label="2048 HU")
    ax.plot(bins, all_n, "k", lw=2, label="Naive")
    ax.plot(bins, ideal, "k--", label="Ideal")
    ax.plot([TPREFERRED, TPREFERRED], ax.get_ylim(), 'C4--')
    ax.set_ylim(0)
    ax.legend()
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Temperature")
    sns.despine(fig, ax)


if __name__ == "__main__":
    # plot training progress
    plot_squared_losses()
    plot_rank_losses()
    # load training data for scaling
    tdata = GradientData.load("gd_training_data.hdf5")
    # plot radial sim results
    plot_sim(tdata, "r")
