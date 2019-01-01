#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script for panels of Figure S1 (Zebrafish network evolution example)
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pl
import seaborn as sns
from Figure3 import mpath
import h5py
from global_defs import GlobalDefs
import core as c
from mo_types import MoTypes

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
    # Panel: WN Playback stimulus response comparison
    # load zebrafish data
    dfile = h5py.File("fish_wn_playback_data.hdf5", 'r')
    padded_temp_input = np.array(dfile["padded_temp_input"])  # in C - needs to be standardized for network!
    model_framerate = np.array(dfile["model_framerate"])
    fish_bout_freq = np.array(dfile["fish_bout_freq"])
    fish_bout_freq_se = np.array(dfile["fish_bout_freq_se"])
    fish_mags = np.array(dfile["fish_mags"])
    fish_mags_se = np.array(dfile["fish_mags_se"])
    pd_stim_seconds = padded_temp_input.size // model_framerate
    stim_seconds = fish_bout_freq.size // model_framerate
    # create time vectors for padded and non-padded data
    model_time = np.linspace(0, stim_seconds, fish_bout_freq.size)
    net_time = np.linspace(0, stim_seconds, stim_seconds*GlobalDefs.frame_rate)
    pd_model_time = np.linspace(0, pd_stim_seconds, padded_temp_input.size)
    pd_net_time = np.linspace(0, pd_stim_seconds, pd_stim_seconds*GlobalDefs.frame_rate)
    pti_network = np.interp(pd_net_time, pd_model_time, padded_temp_input)  # simulation input with initial padding
    std_zf = c.GradientData.load_standards("gd_training_data.hdf5")
    net_in = (pti_network - std_zf.temp_mean) / std_zf.temp_std
    mo = MoTypes(False)
    n_reps = 1000  # run 1000 simulations per network
    net_bout_freqs = []
    net_mags = []
    for p in paths_512_zf:
        m_path = mpath(base_path_zf, p)
        mdata = c.ModelData(m_path)
        gpn_wn = mo.network_model()
        gpn_wn.load(mdata.ModelDefinition, mdata.LastCheckpoint)
        wna = mo.wn_sim(std_zf, gpn_wn, t_preferred=GlobalDefs.tPreferred)
        ev_path = m_path + '/evolve/generation_weights.npy'
        weights = np.load(ev_path)
        w = np.mean(weights[-1, :, :], 0)
        wna.bf_weights = w
        behav_traces = []
        for i in range(n_reps):
            bt = wna.compute_openloop_behavior(net_in)
            behav_traces.append(bt[-net_time.size:])
        behav_traces = np.vstack(behav_traces)
        # compute bout frequency as average of straight/left/right movements
        net_bout_freqs.append(np.mean(behav_traces > 0, 0))
        # compute magnitude by using expected values for straight and turn bouts
        behav_traces[behav_traces < 1] = np.nan
        behav_traces[behav_traces == 1] = 0
        behav_traces[behav_traces > 1] = 30
        net_mags.append(np.nanmean(behav_traces, 0))
