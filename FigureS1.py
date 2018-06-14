#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script for panels of Figure 3 (Zebrafish network ablations)
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
from data_stores import SimulationStore
from Figure3 import mpath

# file definitions
base_path_zf = "./model_data/Adam_1e-4/sepInput_mixTrain/"
paths_512_zf = [f + '/' for f in os.listdir(base_path_zf) if "_3m512_" in f]


if __name__ == "__main__":
    save_folder = "./DataFigures/FigureS1/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42

    # Example evolution on one network
    p = mpath(base_path_zf, paths_512_zf[0])
    evol_p = p + "/evolve/"
    errors = np.load(evol_p + "generation_errors.npy")
    weights = np.load(evol_p + "generation_weights.npy")
    # Panel: Error progression
    fig, ax = pl.subplots()
    ax.errorbar(np.arange(50), np.mean(errors, 1), np.std(errors, 1), linestyle='None', marker='o', color="C1")
    ax.errorbar(49, np.mean(errors, 1)[49], np.std(errors, 1)[49], linestyle='None', marker='o', color="C0")
    ax.errorbar(7, np.mean(errors, 1)[7], np.std(errors, 1)[7], linestyle='None', marker='o', color=(.5, .5, .5))
    ax.set_xlabel("Generation")
    ax.set_ylabel("Navigation error [C]")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "network_0_evolveError.pdf", type="pdf")
    # Panel: Pairwise weight correlations
    corr_0 = []
    corr_7 = []
    corr_49 = []
    for i in range(512):
        for j in range(512):
            if i < j:
                corr_0.append(np.corrcoef(weights[0, i, :], weights[0, j, :])[0, 1])
                corr_7.append(np.corrcoef(weights[7, i, :], weights[7, j, :])[0, 1])
                corr_49.append(np.corrcoef(weights[49, i, :], weights[49, j, :])[0, 1])
    fig, ax = pl.subplots()
    sns.kdeplot(corr_0, ax=ax, color="C1")
    sns.kdeplot(corr_7, ax=ax, color=(.5, .5, .5))
    sns.kdeplot(corr_49, ax=ax, color="C0")
    ax.set_xlabel("Pairwise weight vector correlations")
    ax.set_ylabel("Density")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "network_0_evolveWeightCorrs.pdf", type="pdf")
    # Panel: Example weight matrices
    fig, axes = pl.subplots(ncols=4)
    sns.heatmap(weights[0, :, :], vmin=-3, vmax=3, center=0, cbar=False, cmap="RdBu_r", ax=axes[0], xticklabels=False,
                yticklabels=False, rasterized=True)
    sns.heatmap(weights[7, :, :], vmin=-3, vmax=3, center=0, cbar=False, cmap="RdBu_r", ax=axes[1], xticklabels=False,
                yticklabels=False, rasterized=True)
    sns.heatmap(weights[49, :, :], vmin=-3, vmax=3, center=0, cbar=True, cmap="RdBu_r", ax=axes[2], xticklabels=False,
                yticklabels=False, cbar_ax=axes[3], rasterized=True)
    axes[0].set_ylabel("Generation weight vectors")
    for a in axes[:-1]:
        a.set_xlabel("Weights")
    fig.savefig(save_folder + "network_0_evolveWeights.pdf", type="pdf")
