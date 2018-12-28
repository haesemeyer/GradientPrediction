#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script for panels displaying rl network performance and representation
These will be integrated into different paper figures
"""


import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pl
import seaborn as sns
from Figure3 import mpath
import core as c
import analysis as a
import h5py
from global_defs import GlobalDefs
from scipy.signal import convolve
from RL_trainingGround import CircleRLTrainer
from mo_types import MoTypes
from scipy.ndimage import gaussian_filter1d
from pandas import DataFrame


# file definitions
base_path_rl = "./model_data/SimpleRL_Net/"
paths_rl = [f + '/' for f in os.listdir(base_path_rl) if "mx_" in f]

base_path_zf = "./model_data/Adam_1e-4/sepInput_mixTrain/"
paths_512_zf = [f + '/' for f in os.listdir(base_path_zf) if "_3m512_" in f]


if __name__ == "__main__":
    save_folder = "./DataFigures/FigureRL/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42

    # compute expected temperature mean and standard deviation - the same as during training
    ex1 = CircleRLTrainer(None, 100, 22, 37, 26)
    tm1 = ex1.t_mean
    s1 = ex1.t_std
    ex2 = CircleRLTrainer(None, 100, 14, 29, 26)
    tm2 = ex2.t_mean
    s2 = ex2.t_std
    t_mean = (tm1 + tm2) / 2
    t_std = (s1 + s2) / 2

    # Plot gradient errors during training
    rewards_given = []
    grad_errors = []
    for p in paths_rl:
        l_file = h5py.File(mpath(base_path_rl, p) + "/losses.hdf5", 'r')
        rewards_given.append(np.array(l_file["rewards_given"]))
        grad_errors.append(np.array(l_file["ep_avg_grad_error"]))
        l_file.close()
    # find max reward number
    max_reward = max([np.cumsum(rg)[-1] for rg in rewards_given])
    ip_given = np.arange(max_reward, step=5000)
    grad_errors = np.vstack([np.interp(ip_given, np.cumsum(rg), gaussian_filter1d(ge, 10)) for (rg, ge)
                             in zip(rewards_given, grad_errors)])

    fig, ax = pl.subplots()
    sns.tsplot(grad_errors, ip_given, ax=ax)
    ax.plot([786680, 786680], [2.0, 5.0], 'k--')
    ax.plot([786680, 1e7], [2.4, 2.4], 'k--')
    ax.set_xlabel("# Rewards given")
    ax.set_ylabel("Navigation error")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "rl_training_errors.pdf", type="pdf")

    # Panel - gradient navigation performance - rl net and predictive network
    sim_radius = 100
    sim_min = 22
    sim_max = 37
    bns = np.linspace(0, sim_radius, 100)
    centers = bns[:-1] + np.diff(bns)
    centers = centers / sim_radius * (sim_max - sim_min) + sim_min
    naive_rl = np.zeros((20, centers.size))
    trained_rl = np.zeros_like(naive_rl)
    for i, p in enumerate(paths_rl):
        mdata = c.ModelData(mpath(base_path_rl, p))
        with c.SimpleRLNetwork() as rl_net:
            rl_net.load(mdata.ModelDefinition, mdata.FirstCheckpoint)
            circ_train = CircleRLTrainer(rl_net, sim_radius, sim_min, sim_max, 26)
            circ_train.t_std = t_std
            circ_train.t_mean = t_mean
            naive_pos, _ = circ_train.run_sim(GlobalDefs.n_steps, False)
            rl_net.load(mdata.ModelDefinition, mdata.LastCheckpoint)
            circ_train = CircleRLTrainer(rl_net, sim_radius, sim_min, sim_max, 26)
            circ_train.t_std = t_std
            circ_train.t_mean = t_mean
            trained_pos, _ = circ_train.run_sim(GlobalDefs.n_steps, False)
        naive_rl[i, :] = a.bin_simulation(naive_pos, bns, 'r')
        trained_rl[i, :] = a.bin_simulation(trained_pos, bns, 'r')
    std_zf = c.GradientData.load_standards("gd_training_data.hdf5")
    ana = a.Analyzer(MoTypes(False), std_zf, "sim_store.hdf5", None)
    trained_zf = np.zeros_like(naive_rl)
    for i, p in enumerate(paths_512_zf):
        pos_t = ana.run_simulation(mpath(base_path_zf, p), "r", "trained")
        trained_zf[i, :] = a.bin_simulation(pos_t, bns, 'r')
    fig, ax = pl.subplots()
    sns.tsplot(naive_rl, centers, condition="Naive", color='k')
    sns.tsplot(trained_rl, centers, condition="Trained RL model", color='C1')
    ax.plot(centers, np.mean(trained_zf, 0), color=(0.5, 0.5, 0.5), label="Trained predictive model")
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel("Proportion")
    ax.legend()
    sns.despine(fig, ax)
    fig.savefig(save_folder + "rl_gradient_distribution.pdf", type="pdf")

    # Clustering of temperature responses

    # load and interpolate temperature stimulus
    dfile = h5py.File("stimFile.hdf5", 'r')
    tsin = np.array(dfile['sine_L_H_temp'])
    x = np.arange(tsin.size)  # stored at 20 Hz !
    xinterp = np.linspace(0, tsin.size, tsin.size * GlobalDefs.frame_rate // 20)
    temperature = np.interp(xinterp, x, tsin)
    dfile.close()

    all_cells_rl = []
    p_turn = []
    all_ids_rl = []
    for i, p in enumerate(paths_rl):
        mdata = c.ModelData(mpath(base_path_rl, p))
        with c.SimpleRLNetwork() as rl_net:
            rl_net.load(mdata.ModelDefinition, mdata.LastCheckpoint)
            # prepend lead-in to stimulus
            lead_in = np.full(rl_net.input_dims[2] - 1, np.mean(temperature[:10]))
            temp = np.r_[lead_in, temperature]
            activity_out = rl_net.unit_stimulus_responses(temp, t_mean, t_std)
            cell_res = np.hstack(activity_out['t'])
            pt = activity_out['o'][0][:, 1]
            id_mat = np.zeros((3, cell_res.shape[1]), dtype=np.int32)
            id_mat[0, :] = i
            hidden_sizes = [128] * 3  # currently hard coded, could be extracted from rl_net object
            start = 0
            for layer, hs in enumerate(hidden_sizes):
                id_mat[1, start:start + hs] = layer
                id_mat[2, start:start + hs] = np.arange(hs, dtype=np.int32)
                start += hs
        all_cells_rl.append(cell_res)
        all_ids_rl.append(id_mat)
        p_turn.append(pt[:, None])
    all_cells_rl = np.hstack(all_cells_rl)
    all_ids_rl = np.hstack(all_ids_rl)
    p_turn = np.hstack(p_turn)

    # convolve activity with nuclear gcamp calcium kernel
    raw_activity = all_cells_rl.copy()
    tau_on = 1.4  # seconds
    tau_on *= GlobalDefs.frame_rate  # in frames
    tau_off = 2  # seconds
    tau_off *= GlobalDefs.frame_rate  # in frames
    kframes = np.arange(10 * GlobalDefs.frame_rate)  # 10 s long kernel
    kernel = 2 ** (-kframes / tau_off) * (1 - 2 ** (-kframes / tau_on))
    kernel = kernel / kernel.sum()
    # convolve with our kernel
    for i in range(all_cells_rl.shape[1]):
        all_cells_rl[:, i] = convolve(all_cells_rl[:, i], kernel, mode='full')[:all_cells_rl.shape[0]]

    # cluster data
    clust_ids_rl = a.cluster_activity(8, all_cells_rl, None)[0]

    # plot
    pal = sns.color_palette()  # the default matplotlib color cycle
    plot_cols_rl = {0: pal[0], 1: pal[1], 2: pal[2], 3: pal[3], 4: pal[4], 5: pal[5],
                    6: pal[6], 7: pal[7], -1: (0.6, 0.6, 0.6)}
    n_regs_rl = np.unique(clust_ids_rl).size - 1
    cluster_acts_rl = np.zeros((all_cells_rl.shape[0] // 3, n_regs_rl))
    is_on = np.zeros(n_regs_rl, dtype=bool)
    ax_ix = np.full(n_regs_rl, -1, dtype=int)
    on_count = 0
    off_count = 0
    for i in range(n_regs_rl):
        act = np.mean(a.trial_average(all_cells_rl[:, clust_ids_rl == i], 3), 1)
        cluster_acts_rl[:, i] = act
        is_on[i] = np.corrcoef(act, temperature[:act.size])[0, 1] > 0
        # correspondin axis on ON plot is simply set by order of cluster occurence
        if is_on[i]:
            ax_ix[i] = 0 if on_count < 2 else 1
            on_count += 1
        else:
            ax_ix[i] = 0 if off_count < 2 else 1
            off_count += 1

    fig, (axes_on, axes_off) = pl.subplots(ncols=2, nrows=2, sharex=True)
    time = np.arange(cluster_acts_rl.shape[0]) / GlobalDefs.frame_rate
    for i in range(n_regs_rl):
        act = cluster_acts_rl[:, i].copy()
        act /= np.mean(act[:500])  # normalize to baseline activity which fluctuates a lot in these networks
        if not is_on[i]:
            ax_off = axes_off[ax_ix[i]]
            ax_off.plot(time, act, color=plot_cols_rl[i])
        else:
            ax_on = axes_on[ax_ix[i]]
            ax_on.plot(time, act, color=plot_cols_rl[i])
    axes_off[0].set_xticks([0, 30, 60, 90, 120, 150])
    axes_off[1].set_xticks([0, 30, 60, 90, 120, 150])
    axes_off[0].set_xlabel("Time [s]")
    axes_off[1].set_xlabel("Time [s]")
    axes_on[0].set_ylabel("Cluster average activation")
    axes_off[0].set_ylabel("Cluster average activation")
    sns.despine()
    fig.tight_layout()
    fig.savefig(save_folder + "rl_all_cluster_averages.pdf", type="pdf")

    # plot counts of different clusters
    cl_type_d = {"Fraction": [], "net_id": [], "Cluster ID": []}
    for i in range(len(paths_512_zf)):
        for j in range(-1, 8):
            cl_type_d["Fraction"].append(np.sum(clust_ids_rl == j) / clust_ids_rl.size)
            cl_type_d["net_id"].append(i)
            cl_type_d["Cluster ID"].append(j)
    cl_type_df = DataFrame(cl_type_d)
    fig, ax = pl.subplots()
    sns.barplot("Cluster ID", "Fraction", data=cl_type_df, order=list(range(8)) + [-1],
                ci=68, ax=ax, palette=plot_cols_rl)
    sns.despine(fig)
    fig.savefig(save_folder + "rl_all_cluster_counts.pdf", type="pdf")

    # compare matching of RL and predictive zebrafish types
    ana_zf = a.Analyzer(MoTypes(False), std_zf, "sim_store.hdf5", "activity_store.hdf5")
    all_ids_zf = []
    all_cells_zf = []
    for i, p in enumerate(paths_512_zf):
        cell_res, ids = ana_zf.temperature_activity(mpath(base_path_zf, p), temperature, i)
        all_ids_zf.append(ids)
        all_cells_zf.append(cell_res)
    all_ids_zf = np.hstack(all_ids_zf)
    all_cells_zf = np.hstack(all_cells_zf)
    # convolve with our kernel
    for i in range(all_cells_zf.shape[1]):
        all_cells_zf[:, i] = convolve(all_cells_zf[:, i], kernel, mode='full')[:all_cells_zf.shape[0]]
    clust_ids_zf = a.cluster_activity(8, all_cells_zf, "cluster_info.hdf5")[0]
    cluster_acts_zf = np.zeros((all_cells_rl.shape[0] // 3, n_regs_rl))
    for i in range(n_regs_rl):
        act = np.mean(a.trial_average(all_cells_zf[:, clust_ids_zf == i], 3), 1)
        cluster_acts_zf[:, i] = act

    corr_mat = np.zeros((cluster_acts_zf.shape[1], cluster_acts_rl.shape[1]))
    for i in range(cluster_acts_zf.shape[1]):
        for j in range(cluster_acts_rl.shape[1]):
            corr_mat[i, j] = np.corrcoef(cluster_acts_zf[:, i], cluster_acts_rl[:, j])[0, 1]

    fig, ax = pl.subplots()
    names = ['0', 'Fast OFF', 'Int OFF', 'Slow OFF', 'Fast ON', 'Slow ON', '6', '7']
    sns.heatmap(np.round(corr_mat, 2), vmin=-1, vmax=1, center=0, cmap="RdBu_r", annot=True, yticklabels=names)
    ax.set_xlabel("RL Network")
    ax.set_ylabel("Predictive Network")
    fig.tight_layout()
    fig.savefig(save_folder + "rl_cluster_fish_correlations.pdf", type="pdf")
