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


def loss_file(path):
    return base_path + path + "losses.hdf5"


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


if __name__ == "__main__":
    # plot training progress
    plot_squared_losses()
    plot_rank_losses()
