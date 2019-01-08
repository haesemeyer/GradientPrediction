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
from multiprocessing import Pool
from pandas import DataFrame
from scipy.stats import wilcoxon

# file definitions
base_path_zf = "./model_data/Adam_1e-4/sepInput_mixTrain/"
paths_512_zf = [f + '/' for f in os.listdir(base_path_zf) if "_3m512_" in f]


def playback_response_helper(mo_type, model_path, stimulus, std, no_pad_size, nreps):
    mdata = c.ModelData(model_path)
    gpn_wn = mo_type.network_model()
    gpn_wn.load(mdata.ModelDefinition, mdata.LastCheckpoint)
    wna = mo_type.wn_sim(std, gpn_wn, t_preferred=GlobalDefs.tPreferred)
    ev_path = model_path + '/evolve/generation_weights.npy'
    ev_weights = np.load(ev_path)
    w = np.mean(ev_weights[-1, :, :], 0)
    wna.bf_weights = w
    wna.eval_every = 2  # fast evaluation to keep up with stimulus fluctuations
    traces = []
    for r in range(nreps):
        bt = wna.compute_openloop_behavior(stimulus)[0].astype(float)
        traces.append(bt[-no_pad_size:])
    traces = np.vstack(traces)
    p_move = np.mean(traces > 0, 0)  # probability of selecting a movement bout (p_bout weights plus pred. control)
    p_bout = np.mean(traces > -1, 0)  # any behavior selected - purely under control of p_bout weights
    # compute magnitude by using expected values for straight and turn bouts
    traces[traces < 1] = np.nan
    traces[traces < 2] = 0
    traces[traces > 1] = 30
    mag = np.nanmean(traces, 0)
    return p_move, p_bout, mag


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
    model_framerate = np.array(dfile["model_framerate"])[0]
    fish_bout_freq = np.array(dfile["fish_bout_freq"])
    fish_bout_freq_se = np.array(dfile["fish_bout_freq_se"])
    fish_mags = np.array(dfile["mags_by_fish"])
    dfile.close()
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
    n_reps = 5000  # run 5000 simulations per network
    net_bout_freqs = []
    net_p_bout_freqs = []
    net_mags = []
    process_pool = Pool(processes=4)
    process_ar = []
    for p in paths_512_zf:
        m_path = mpath(base_path_zf, p)
        process_ar.append(process_pool.apply_async(playback_response_helper,
                                                   [mo, m_path, net_in, std_zf, net_time.size, n_reps]))
    for i, ar in enumerate(process_ar):
        bf, pb, m = ar.get()
        net_bout_freqs.append(bf * GlobalDefs.frame_rate)
        net_p_bout_freqs.append(pb * GlobalDefs.frame_rate)
        net_mags.append(m)
        print("Process {0} of {1} completed".format(i+1, len(paths_512_zf)))
    process_pool.close()
    net_bout_freqs = np.vstack(net_bout_freqs)
    net_p_bout_freqs = np.vstack(net_p_bout_freqs)
    net_mags = np.vstack(net_mags)
    # interpolate fish-data to net timebase
    fish_bout_freq = np.interp(net_time, model_time, fish_bout_freq)
    fish_bout_freq_se = np.interp(net_time, model_time, fish_bout_freq_se)
    # interpolate magnitudes for each fish
    fish_mags_interp = np.zeros((fish_mags.shape[0], fish_bout_freq.size))
    for i, fm in enumerate(fish_mags):
        fish_mags_interp[i, :] = np.interp(net_time, model_time, fm)
    # plot stimulus
    fig, ax = pl.subplots()
    ax.plot(net_time[:-4], net_in[-net_time.size:-4]*std_zf.temp_std + std_zf.temp_mean, 'k')
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [C]")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "playback_stimulus.pdf", type="pdf")
    # plot fish and model bout frequency across time
    corr_bout = np.corrcoef(fish_bout_freq[:-4], np.nanmean(net_bout_freqs, 0)[:-4])[0, 1]
    corr_p_bout = np.corrcoef(fish_bout_freq[:-4], np.nanmean(net_p_bout_freqs, 0)[:-4])[0, 1]
    fig, ax = pl.subplots()
    sns.tsplot(net_bout_freqs[:, :-4], net_time[:-4], condition="r = {0}".format(np.round(corr_bout, 2)), color="C1")
    ax.plot(net_time[:-4], np.mean(net_p_bout_freqs, 0)[:-4], color=(0.6, 0.6, 0.6), lw=0.5,
            label="r = {0}".format(np.round(corr_p_bout, 2)))
    ax.fill_between(net_time[:-4], fish_bout_freq[:-4]-fish_bout_freq_se[:-4],
                    fish_bout_freq[:-4]+fish_bout_freq_se[:-4], facecolor='C0', alpha=0.5)
    ax.plot(net_time[:-4], fish_bout_freq[:-4], 'C0')
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Swim frequency [Hz]")
    ax.legend()
    sns.despine(fig, ax)
    fig.savefig(save_folder + "playback_bfreq_comparison.pdf", type="pdf")
    # plot average turn frequency in rising and falling phases
    dtemp = np.diff(net_in)[-net_time.size:]
    rising = dtemp > 0.005
    falling = dtemp < -0.005
    data_dict = {"Type": [], "Direction": [], "TurnMagnitude": []}
    for fish in fish_mags_interp:
        data_dict["Type"].append("zebrafish")
        data_dict["Direction"].append("rising")
        data_dict["TurnMagnitude"].append(np.nanmean(fish[rising]))
        data_dict["Type"].append("zebrafish")
        data_dict["Direction"].append("falling")
        data_dict["TurnMagnitude"].append(np.nanmean(fish[falling]))
    for net in net_mags:
        data_dict["Type"].append("Network")
        data_dict["Direction"].append("rising")
        data_dict["TurnMagnitude"].append(np.nanmean(net[rising]))
        data_dict["Type"].append("Network")
        data_dict["Direction"].append("falling")
        data_dict["TurnMagnitude"].append(np.nanmean(net[falling]))
    data_frame = DataFrame(data_dict)
    fig, ax = pl.subplots()
    sns.barplot("Type", "TurnMagnitude", "Direction", data_frame, ["zebrafish", "Network"], ["falling", "rising"],
                ci=68)
    sns.despine(fig, ax)
    fig.savefig(save_folder + "playback_turn_modulation.pdf", type="pdf")
    # compute statistics
    res_fish = wilcoxon(
        data_frame.TurnMagnitude[np.logical_and(data_frame.Type == "zebrafish", data_frame.Direction == "falling")],
        data_frame.TurnMagnitude[np.logical_and(data_frame.Type == "zebrafish", data_frame.Direction == "rising")])
    res_network = wilcoxon(
        data_frame.TurnMagnitude[np.logical_and(data_frame.Type == "Network", data_frame.Direction == "falling")],
        data_frame.TurnMagnitude[np.logical_and(data_frame.Type == "Network", data_frame.Direction == "rising")])
    print("Fish comparison. Wilcoxon statistic {0}; p-value {1}".format(res_fish[0], res_fish[1]))
    print("Network comparison. Wilcoxon statistic {0}; p-value {1}".format(res_network[0], res_network[1]))
