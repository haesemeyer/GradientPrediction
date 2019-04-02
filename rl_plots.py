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
from Figure1 import get_bout_da, get_bout_starts
from scipy.stats import wilcoxon
import pickle
from zfish_ann_correspondence import RegionResults, create_corr_mat, greedy_max_clust
from data_stores import SimulationStore


# file definitions
base_path_rl = "./model_data/FullRL_Net/"
paths_rl = [f + '/' for f in os.listdir(base_path_rl) if "mx_disc_" in f]

base_path_zf = "./model_data/Adam_1e-4/sepInput_mixTrain/"
paths_512_zf = [f + '/' for f in os.listdir(base_path_zf) if "_3m512_" in f]


def run_flat_gradient(model_path):
    mdata = c.ModelData(model_path)
    gpn = c.SimpleRLNetwork()
    gpn.load(mdata.ModelDefinition, mdata.LastCheckpoint)
    arena = CircleRLTrainer(rl_net, sim_radius, 22, 22, 26)
    arena.t_std = t_std
    arena.t_mean = t_mean
    arena.p_explore = 0.25
    return circ_train.run_sim(GlobalDefs.n_steps, False)[0]


def compute_da_modulation(model_path, pos_trained):
    pos_flt = run_flat_gradient(model_path)
    bs_ev = get_bout_starts(pos_trained)
    bs_flt = get_bout_starts(pos_flt)
    # get delta angle of each bout
    da_ev = get_bout_da(pos_trained, bs_ev)
    da_flt = get_bout_da(pos_flt, bs_flt)
    # get temperature at each bout start
    temp_ev = a.temp_convert(np.sqrt(np.sum(pos_trained[bs_ev.astype(bool), :2]**2, 1)), 'r')
    temp_flt = a.temp_convert(np.sqrt(np.sum(pos_flt[bs_flt.astype(bool), :2] ** 2, 1)), 'r')
    # get delta-temperature effected by each bout
    dt_ev = np.r_[0, np.diff(temp_ev)]
    dt_flt = np.r_[0, np.diff(temp_flt)]
    # only consider data above T_Preferred and away from the edge
    valid_ev = np.logical_and(temp_ev > GlobalDefs.tPreferred, temp_ev < GlobalDefs.circle_sim_params["t_max"]-1)
    valid_flt = np.logical_and(temp_flt > GlobalDefs.tPreferred, temp_flt < GlobalDefs.circle_sim_params["t_max"] - 1)
    da_ev = da_ev[valid_ev]
    da_flt = da_flt[valid_flt]
    dt_ev = dt_ev[valid_ev]
    dt_flt = dt_flt[valid_flt]
    # get turn magnitude for up and down gradient
    up_grad_ev = np.mean(np.abs(da_ev[dt_ev > 0.5]))
    dn_grad_ev = np.mean(np.abs(da_ev[dt_ev < -0.5]))
    up_grad_flt = np.mean(np.abs(da_flt[dt_flt > 0.5]))
    dn_grad_flt = np.mean(np.abs(da_flt[dt_flt < -0.5]))
    up_change = up_grad_ev / up_grad_flt
    dn_change = dn_grad_ev / dn_grad_flt
    return dn_change, up_change


def compute_gradient_bout_frequency(positions):
    def bout_freq(pos: np.ndarray):
        r = np.sqrt(np.sum(pos[:, :2]**2, 1))  # radial position
        bs = get_bout_starts(pos)  # bout starts
        bins = np.linspace(0, GlobalDefs.circle_sim_params["radius"], 6)
        bcenters = bins[:-1] + np.diff(bins)/2
        cnt_r = np.histogram(r, bins)[0]
        cnt_r_bs = np.histogram(r[bs > 0.1], bins)[0]
        bfreq = cnt_r_bs / cnt_r * GlobalDefs.frame_rate
        return bfreq, bcenters

    bf, bc = bout_freq(positions)
    return bc, bf


def compute_da_coherence(model_path, drop_list=None):
    with SimulationStore("sim_store.hdf5", std_zf, MoTypes(False)) as sim_store:
        pos_ev = sim_store.get_sim_pos(model_path, "r", "bfevolve", drop_list)
    bs_ev = get_bout_starts(pos_ev)
    # get delta angle of each bout
    da_ev = np.rad2deg(get_bout_da(pos_ev, bs_ev))
    # convert into appproximation of S, L and R behaviors
    bhv_ev = np.ones_like(da_ev)
    bhv_ev[da_ev < -10] = 2
    bhv_ev[da_ev > 10] = 3
    return a.turn_coherence(bhv_ev, 10), da_ev


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
        try:
            l_file = h5py.File(mpath(base_path_rl, p) + "/losses.hdf5", 'r')
        except IOError:
            continue
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
    ax.plot([786680, 1.7e7], [2.4, 2.4], 'k--')
    ax.set_xlabel("# Rewards given")
    ax.set_ylabel("Navigation error")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "rl_training_errors.pdf", type="pdf")

    # store bout frequency modulation in gradients
    bout_freqs = np.empty((len(paths_rl), 5))
    bout_bin_centers = None
    # store turn angle modulation during navigation
    turn_mod_data_dict = {"Type": [], "Direction": [], "Turn Enhancement": []}
    net_down = []
    net_up = []
    # store turn persistence
    rl_turn_coherence = []
    # store turn angles
    rl_da = []
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
        with c.ReinforcementLearningNetwork() as rl_net:
            rl_net.load(mdata.ModelDefinition, mdata.FirstCheckpoint)
            circ_train = CircleRLTrainer(rl_net, sim_radius, sim_min, sim_max, 26)
            circ_train.t_std = t_std
            circ_train.t_mean = t_mean
            circ_train.p_explore = 0.25  # try to match to exploration in predictive network (best taken 50% of time)
            naive_pos = circ_train.run_sim(GlobalDefs.n_steps, False)[0]
            rl_net.load(mdata.ModelDefinition, mdata.LastCheckpoint)
            circ_train = CircleRLTrainer(rl_net, sim_radius, sim_min, sim_max, 26)
            circ_train.t_std = t_std
            circ_train.t_mean = t_mean
            circ_train.p_explore = 0.25  # try to match to exploration in predictive network (best taken 50% of time)
            trained_pos, _, trained_behav = circ_train.run_sim(GlobalDefs.n_steps, False)
            rl_turn_coherence.append(a.turn_coherence(np.array(trained_behav), 10))
            rl_da.append(np.rad2deg(get_bout_da(trained_pos, get_bout_starts(trained_pos))))
            # process bout frequency
            bout_bin_centers, bfreqs = compute_gradient_bout_frequency(trained_pos)
            bout_freqs[i, :] = bfreqs
            # process turn modulation
            dn, up = compute_da_modulation(mpath(base_path_rl, p), trained_pos)
            net_down.append(dn)
            net_up.append(up)
            turn_mod_data_dict["Type"] += ["RL Network", "RL Network"]
            turn_mod_data_dict["Direction"] += ["down", "up"]
            turn_mod_data_dict["Turn Enhancement"] += [dn, up]
        naive_rl[i, :] = a.bin_simulation(naive_pos, bns, 'r')
        trained_rl[i, :] = a.bin_simulation(trained_pos, bns, 'r')

    bout_bin_centers = bout_bin_centers / sim_radius * (sim_max - sim_min) + sim_min
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

    # plot bout frequency modulation in the gradient
    fig, ax = pl.subplots()
    sns.tsplot(bout_freqs, bout_bin_centers, n_boot=1000, color="C1", err_style="ci_band", estimator=np.nanmean)
    ax.set_xlim(23, 36)
    ax.set_xticks([25, 30, 35])
    ax.set_yticks([0.5, 0.75, 1, 1.25])
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel("Swim frequency [Hz]")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "rl_net_gradient_swim_frequency.pdf", type="pdf")
    # plot turn modulation
    data_frame = DataFrame(turn_mod_data_dict)
    fig, ax = pl.subplots()
    sns.barplot("Type", "Turn Enhancement", "Direction", data_frame, ["RL Network"], ["down", "up"], ci=68)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    sns.despine(fig, ax)
    fig.savefig(save_folder + "rl_net_grad_turn_modulation.pdf", type="pdf")
    print("Network comparison. Wilcoxon statistic {0}; p-value {1}".format(*wilcoxon(net_down, net_up)))
    # plot turn coherence
    all_bout_da = np.load("all_bout_da.npy")  # zebrafish gradient data
    all_bout_phases = np.load("all_bout_phases.npy")
    pd_turn_coherence = []
    pd_da = []
    for i, p in enumerate(paths_512_zf):
        tc, da = compute_da_coherence(mpath(base_path_zf, p))
        pd_turn_coherence.append(tc)
        pd_da.append(da)
    zf_turn_coherence, zf_da = [], []
    for ar, bp in zip(all_bout_da, all_bout_phases):
        ar = ar[bp == 8]  # limit to gradient phase
        bhv = np.ones_like(ar)
        bhv[ar < -10] = 2
        bhv[ar > 10] = 3
        zf_turn_coherence.append(a.turn_coherence(bhv, 10))
        zf_da.append(ar)
    fig, ax = pl.subplots()
    sns.tsplot(zf_turn_coherence, np.arange(10)+1, color="C0")
    sns.tsplot(rl_turn_coherence, np.arange(10)+1, color="C1")
    sns.tsplot(pd_turn_coherence, np.arange(10)+1, color=[0.5, 0.5, 0.5])
    ax.plot([1, 10], [0.5, 0.5], 'k--')
    ax.set_xlabel("Subsequent turns")
    ax.set_ylabel("p(Same direction)")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "rl_net_gradient_turn_coherence.pdf", type="pdf")
    # plot bout angle distribution based on median angle for fish, predictive and rl network
    fig, ax = pl.subplots()
    for da in zf_da:
        if np.median(da) < 0:
            sns.kdeplot(da, color="C0", alpha=0.5, lw=0.5, ax=ax)
        else:
            sns.kdeplot(da, color="C3", alpha=0.5, lw=0.5, ax=ax)
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    ax.set_xlabel("Turn angle [deg]")
    ax.set_ylabel("Density")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "zebrafish_turn_angles.pdf", type="pdf")
    fig, ax = pl.subplots()
    for da in pd_da:
        if np.median(da) < 0:
            sns.kdeplot(da, color="C0", alpha=0.5, lw=0.5, ax=ax)
        else:
            sns.kdeplot(da, color="C3", alpha=0.5, lw=0.5, ax=ax)
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    ax.set_xlabel("Turn angle [deg]")
    ax.set_ylabel("Density")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "pred_net_turn_angles.pdf", type="pdf")
    fig, ax = pl.subplots()
    for da in rl_da:
        if np.median(da) < 0:
            sns.kdeplot(da, color="C0", alpha=0.5, lw=0.5, ax=ax)
        else:
            sns.kdeplot(da, color="C3", alpha=0.5, lw=0.5, ax=ax)
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    ax.set_xlabel("Turn angle [deg]")
    ax.set_ylabel("Density")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "rl_net_turn_angles.pdf", type="pdf")

    # Clustering of temperature responses
    # load and interpolate temperature stimulus
    dfile = h5py.File("stimFile.hdf5", 'r')
    tsin = np.array(dfile['sine_L_H_temp'])
    x = np.arange(tsin.size)  # stored at 20 Hz !
    xinterp = np.linspace(0, tsin.size, tsin.size * GlobalDefs.frame_rate // 20)
    temperature = np.interp(xinterp, x, tsin)
    dfile.close()

    all_cells_rl = []
    all_ids_rl = []
    for i, p in enumerate(paths_rl):
        mdata = c.ModelData(mpath(base_path_rl, p))
        with c.ReinforcementLearningNetwork() as rl_net:
            rl_net.load(mdata.ModelDefinition, mdata.LastCheckpoint)
            # prepend lead-in to stimulus
            lead_in = np.full(rl_net.input_dims[2] - 1, np.mean(temperature[:10]))
            temp = np.r_[lead_in, temperature]
            activity_out = rl_net.unit_stimulus_responses(temp, t_mean, t_std)
            cell_res = np.hstack(activity_out['t'])
            id_mat = np.zeros((3, cell_res.shape[1]), dtype=np.int32)
            id_mat[0, :] = i
            hidden_sizes = [128] * rl_net.n_layers_branch
            start = 0
            for layer, hs in enumerate(hidden_sizes):
                id_mat[1, start:start + hs] = layer
                id_mat[2, start:start + hs] = np.arange(hs, dtype=np.int32)
                start += hs
        all_cells_rl.append(cell_res)
        all_ids_rl.append(id_mat)
    all_cells_rl = np.hstack(all_cells_rl)
    all_ids_rl = np.hstack(all_ids_rl)

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

    # compare matching of RL and zebrafish types
    # load zebrafish region results and create Rh56 regressor matrix for FastON, SlowON, FastOFF, SlowOFF
    result_labels = ["Rh6"]
    region_results = {}  # type: Dict[str, RegionResults]
    analysis_file = h5py.File('regiondata.hdf5', 'r')
    for rl in result_labels:
        region_results[rl] = pickle.loads(np.array(analysis_file[rl]))
    analysis_file.close()
    rh_56_calcium = region_results["Rh6"].regressors[:, :-1]
    # the names of these regressors according to Haesemeyer et al., 2018
    reg_names = ["Fast ON", "Slow ON", "Fast OFF", "Slow OFF"]
    ca_time = np.linspace(0, 165, rh_56_calcium.shape[0])
    net_time = np.linspace(0, 165, cluster_acts_rl.shape[0])
    zf_cluster_centroids = np.zeros((net_time.size, rh_56_calcium.shape[1]))
    for i in range(rh_56_calcium.shape[1]):
        zf_cluster_centroids[:, i] = np.interp(net_time, ca_time, rh_56_calcium[:, i])

    # perform all pairwise correlations between the network and zebrafish units during sine stimulus phase
    cm_sine = create_corr_mat(cluster_acts_rl, zf_cluster_centroids, net_time, 60, 105)
    assignment = greedy_max_clust(cm_sine, 0.6, reg_names)
    assign_labels = [assignment[k] for k in range(cm_sine.shape[0])]

    # plot correlation matrix
    fig, ax = pl.subplots()
    sns.heatmap(cm_sine, vmin=-1, vmax=1, center=0, annot=True, xticklabels=reg_names, yticklabels=assign_labels, ax=ax,
                cmap="RdBu_r")
    ax.set_xlabel("Zebrafish cell types")
    ax.set_ylabel("ANN clusters")
    fig.tight_layout()
    fig.savefig(save_folder + "ZFish_rlNet_Correspondence.pdf", type="pdf")
