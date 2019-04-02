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
from pandas import DataFrame
from scipy.stats import wilcoxon


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


def get_bout_starts(pos: np.ndarray) -> np.ndarray:
    """
    Extract bout starts from network position trace
    :param pos: nx3 trace of x, y, angle at each timepoint
    :return: Array of indices corresponding to bout starts
    """
    spd = np.r_[0, np.sqrt(np.sum(np.diff(pos[:, :2], axis=0) ** 2, 1))]  # speed
    bs = np.r_[0, np.diff(spd) > 0.00098]  # bout starts
    return bs


def get_bout_da(pos: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """
    For each bout indicated by starts get the angle turned
    :param pos: nx3 trace of x, y, angle at each timepoint
    :param starts: Array of indices corresponding to bout starts
    :return: For each bout in starts the turning angle
    """
    starts = np.arange(pos.shape[0])[starts.astype(bool)]
    ix_pre = starts - 10
    ix_pre[ix_pre < 0] = 0
    ix_post = starts + 10
    ix_post[ix_post >= pos.shape[0]] = pos.shape[0]-1
    da = pos[ix_post, 2] - pos[ix_pre, 2]
    return da


def compute_gradient_bout_frequency(model_path, drop_list=None):
    def bout_freq(pos: np.ndarray):
        r = np.sqrt(np.sum(pos[:, :2]**2, 1))  # radial position
        bs = get_bout_starts(pos)  # bout starts
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


def run_flat_gradient(model_path, drop_list=None):
    mdata = c.ModelData(model_path)
    gpn = MoTypes(False).network_model()
    gpn.load(mdata.ModelDefinition, mdata.LastCheckpoint)
    flt_params = GlobalDefs.circle_sim_params.copy()
    flt_params["t_max"] = flt_params["t_min"]
    sim = MoTypes(False).rad_sim(gpn, std, **flt_params)
    sim.t_max = sim.t_min  # reset gradient to be flat
    sim.remove = drop_list
    evo_path = model_path + '/evolve/generation_weights.npy'
    evo_weights = np.load(evo_path)
    w = np.mean(evo_weights[-1, :, :], 0)
    sim.bf_weights = w
    return sim.run_simulation(GlobalDefs.n_steps, False)


def compute_da_modulation(model_path, drop_list=None):
    with SimulationStore("sim_store.hdf5", std, MoTypes(False)) as sim_store:
        pos_ev = sim_store.get_sim_pos(model_path, "r", "bfevolve", drop_list)
    pos_flt = run_flat_gradient(model_path, drop_list)
    bs_ev = get_bout_starts(pos_ev)
    bs_flt = get_bout_starts(pos_flt)
    # get delta angle of each bout
    da_ev = get_bout_da(pos_ev, bs_ev)
    da_flt = get_bout_da(pos_flt, bs_flt)
    # get temperature at each bout start
    temp_ev = a.temp_convert(np.sqrt(np.sum(pos_ev[bs_ev.astype(bool), :2]**2, 1)), 'r')
    temp_flt = a.temp_convert(np.sqrt(np.sum(pos_flt[bs_flt.astype(bool), :2] ** 2, 1)), 'r')
    # get delta-temperature effected by each previous bout
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

    # fifth panel - turning modulation by previous bout direction
    # load real fish gradient data
    dfile = h5py.File("fish_rgrad_data.hdf5", 'r')
    fish_down = np.array(dfile["dn_change"])
    fish_up = np.array(dfile["up_change"])
    dfile.close()
    data_dict = {"Type": [], "Direction": [], "Turn Enhancement": []}
    for fd in fish_down:
        data_dict["Type"].append("zebrafish")
        data_dict["Direction"].append("down")
        data_dict["Turn Enhancement"].append(fd)
    for fu in fish_up:
        data_dict["Type"].append("zebrafish")
        data_dict["Direction"].append("up")
        data_dict["Turn Enhancement"].append(fu)
    # compute same quantities across our networks
    net_down = []
    net_up = []
    for p in paths_512:
        dn, up = compute_da_modulation(mpath(p))
        net_down.append(dn)
        net_up.append(up)
        data_dict["Type"] += ["Network", "Network"]
        data_dict["Direction"] += ["down", "up"]
        data_dict["Turn Enhancement"] += [dn, up]
    data_frame = DataFrame(data_dict)
    fig, ax = pl.subplots()
    sns.barplot("Type", "Turn Enhancement", "Direction", data_frame, ["zebrafish", "Network"], ["down", "up"], ci=68)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    sns.despine(fig, ax)
    fig.savefig(save_folder + "grad_turn_modulation.pdf", type="pdf")
    print("Fish comparison. Wilcoxon statistic {0}; p-value {1}".format(*wilcoxon(fish_down, fish_up)))
    print("Network comparison. Wilcoxon statistic {0}; p-value {1}".format(*wilcoxon(net_down, net_up)))
