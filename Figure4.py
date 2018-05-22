#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script for panels of Figure 4 (Phototaxis and C elegans networks w. zfish comparison)
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
from pandas import DataFrame
import pickle
from Figure3 import mpath
from scipy.signal import convolve
from sklearn.decomposition import PCA


# file definitions
base_path_zf = "./model_data/Adam_1e-4/sepInput_mixTrain/"
paths_512_zf = [f + '/' for f in os.listdir(base_path_zf) if "_3m512_" in f]

base_path_ce = "./model_data/CE_Adam_1e-4/"
paths_512_ce = [f + '/' for f in os.listdir(base_path_ce) if "_3m512_" in f]

base_path_pt = "./model_data/Phototaxis/"
paths_512_pt = [f + '/' for f in os.listdir(base_path_pt) if "_3m512_" in f]


def test_loss(base, path):
    fname = base + path + "losses.hdf5"
    lossfile = h5py.File(fname, "r")
    test_losses = np.array(lossfile["test_losses"])
    rank_errors = np.array(lossfile["test_rank_errors"])
    timepoints = np.array(lossfile["test_eval"])
    return timepoints, test_losses, rank_errors


def plot_pc(index, pc_cords, spec_id, explained_variance, prefix: str):
    """
    Plot principal component comparison between to "species" as KDE
    :param index: The index of the pc to compare
    :param pc_cords: Coordinates of points in pca space
    :param spec_id: For each coordinate the species of model it belongs to
    :param explained_variance: The variance explained by this PC for axis normalization
    :param prefix: Figure file prefix
    """
    f, axis = pl.subplots()
    for spid in np.unique(spec_id):
        sns.kdeplot(pc_cords[spec_id == spid, index]/np.sqrt(explained_variance[index]), shade=True, ax=axis)
    axis.set_xlabel("PC {0}".format(index+1))
    axis.set_ylabel("Density")
    sns.despine(f, axis)
    f.savefig(save_folder + prefix + "_PC_SpaceComparison_PC{0}.pdf".format(index+1), type="pdf")


# The following functions are necessary since phototaxis simulations are currently not part of the data store scheme
def do_simulation(path):
    """
    Uses a model identified by path to run a naive and a trained simulation
    :param path: The model path
    :return:
        [0]: The facing angle bin centers
        [1]: The occupancy of the naive model
        [2]: The occupancy of the trained model
    """
    global std_pt
    bins = np.linspace(-np.pi, np.pi, 100)
    # bin-centers in degress
    bcenters = bins[:-1]+np.diff(bins)/2
    # naive simulation
    mdata = c.ModelData(path)
    model_naive = c.ZfGpNetworkModel()
    model_naive.load(mdata.ModelDefinition, mdata.FirstCheckpoint)
    model_trained = c.ZfGpNetworkModel()
    model_trained.load(mdata.ModelDefinition, mdata.LastCheckpoint)
    sim = MoTypes(False).pt_sim(model_naive, std_pt, 100)
    pos_naive = sim.run_simulation(GlobalDefs.n_steps)
    h_naive = a.bin_simulation_pt(pos_naive, bins)
    sim = MoTypes(False).pt_sim(model_trained, std_pt, 100)
    pos_trained = sim.run_simulation(GlobalDefs.n_steps)
    h_trained = a.bin_simulation_pt(pos_trained, bins)
    return bcenters, h_naive, h_trained


def get_cell_responses(path, temp):
    """
    Loads a model and computes the temperature response of all neurons returning response matrix
    :param path: Model path
    :param temp: Temperature stimulus
    :return: n-timepoints x m-neurons matrix of responses
    """
    global std_pt
    mdata = c.ModelData(path)
    # create our model and load from last checkpoint
    gpn = c.ZfGpNetworkModel()
    gpn.load(mdata.ModelDefinition, mdata.LastCheckpoint)
    # prepend lead-in to stimulus
    lead_in = np.full(gpn.input_dims[2] - 1, np.mean(temp[:10]))
    temp = np.r_[lead_in, temp]
    activities = gpn.unit_stimulus_responses(temp, None, None, std_pt)
    return np.hstack(activities['t']) if 't' in activities else np.hstack(activities['m'])


if __name__ == "__main__":
    save_folder = "./DataFigures/Figure4/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42

    # Panel 1 - test error during phototaxis training
    test_time = test_loss(base_path_pt, paths_512_pt[0])[0]
    test_512 = np.vstack([test_loss(base_path_pt, lp)[1] for lp in paths_512_pt])
    fig, ax = pl.subplots()
    sns.tsplot(np.log10(test_512), test_time, ax=ax, color="C1", n_boot=1000, condition="512 HU")
    epoch_times = np.linspace(0, test_time.max(), 10, endpoint=False)
    for e in epoch_times:
        ax.plot([e, e], [-.5, .1], 'k--', lw=0.25)
    ax.set_ylabel("log(Squared test error)")
    ax.set_xlabel("Training step")
    ax.set_xlim(-10000)
    ax.set_xticks([0, 100000, 200000, 300000, 400000])
    ax.legend()
    sns.despine(fig, ax)
    fig.savefig(save_folder+"pt_test_errors.pdf", type="pdf")

    # Panel 5: Test error during C. elegans model training
    test_time = test_loss(base_path_ce, paths_512_ce[0])[0]
    test_512 = np.vstack([test_loss(base_path_ce, lp)[1] for lp in paths_512_ce])
    fig, ax = pl.subplots()
    sns.tsplot(np.log10(test_512), test_time, ax=ax, color="C1", n_boot=1000, condition="512 HU")
    epoch_times = np.linspace(0, test_time.max(), 10, endpoint=False)
    for e in epoch_times:
        ax.plot([e, e], [-1.2, .4], 'k--', lw=0.25)
    ax.set_ylabel("log(Squared test error)")
    ax.set_xlabel("Training step")
    ax.set_xlim(-10000)
    ax.set_xticks([0, 250000, 500000])
    ax.legend()
    sns.despine(fig, ax)
    fig.savefig(save_folder + "ce_test_errors.pdf", type="pdf")

    std_zf = c.GradientData.load_standards("gd_training_data.hdf5")
    ana_zf = a.Analyzer(MoTypes(False), std_zf, "sim_store.hdf5", "activity_store.hdf5")
    std_ce = c.GradientData.load_standards("ce_gd_training_data.hdf5")
    ana_ce = a.Analyzer(MoTypes(True), std_ce, "ce_sim_store.hdf5", "ce_activity_store.hdf5")
    std_pt = c.GradientData.load_standards("photo_training_data.hdf5")

    # load activity clusters from file
    clfile = h5py.File("ce_cluster_info.hdf5", "r")
    clust_ids_ce = np.array(clfile["clust_ids"])
    clfile.close()

    # load and interpolate temperature stimulus
    dfile = h5py.File("stimFile.hdf5", 'r')
    tsin = np.array(dfile['sine_L_H_temp'])
    x = np.arange(tsin.size)  # stored at 20 Hz !
    xinterp = np.linspace(0, tsin.size, tsin.size * GlobalDefs.frame_rate // 20)
    temperature = np.interp(xinterp, x, tsin)
    dfile.close()

    # get activity data - corresponding to sine-wave
    all_ids_zf = []
    all_cells_zf = []
    for i, p in enumerate(paths_512_zf):
        cell_res, ids = ana_zf.temperature_activity(mpath(base_path_zf, p), temperature, i)
        all_ids_zf.append(ids)
        all_cells_zf.append(cell_res)
    all_ids_zf = np.hstack(all_ids_zf)
    all_cells_zf = np.hstack(all_cells_zf)
    all_ids_ce = []
    all_cells_ce = []
    for i, p in enumerate(paths_512_ce):
        cell_res, ids = ana_ce.temperature_activity(mpath(base_path_ce, p), temperature, i)
        all_ids_ce.append(ids)
        all_cells_ce.append(cell_res)
    all_ids_ce = np.hstack(all_ids_ce)
    all_cells_ce = np.hstack(all_cells_ce)
    all_cells_pt = []
    for p in paths_512_pt:
        all_cells_pt.append(get_cell_responses(mpath(base_path_pt, p), temperature))
    all_cells_pt = np.hstack(all_cells_pt)

    # convolve activity with nuclear gcamp calcium kernel
    tau_on = 1.4  # seconds
    tau_on *= GlobalDefs.frame_rate  # in frames
    tau_off = 2  # seconds
    tau_off *= GlobalDefs.frame_rate  # in frames
    kframes = np.arange(10 * GlobalDefs.frame_rate)  # 10 s long kernel
    kernel = 2 ** (-kframes / tau_off) * (1 - 2 ** (-kframes / tau_on))
    kernel = kernel / kernel.sum()
    # convolve with our kernel
    for i in range(all_cells_zf.shape[1]):
        all_cells_zf[:, i] = convolve(all_cells_zf[:, i], kernel, method='full')[:all_cells_zf.shape[0]]
    for i in range(all_cells_ce.shape[1]):
        all_cells_ce[:, i] = convolve(all_cells_ce[:, i], kernel, method='full')[:all_cells_ce.shape[0]]
    for i in range(all_cells_pt.shape[1]):
        all_cells_pt[:, i] = convolve(all_cells_pt[:, i], kernel, method='full')[:all_cells_pt.shape[0]]

    # Panel 2 - naive and trained phototaxis performance
    all_n = []
    t_512 = []
    bins = None
    for p512 in paths_512_pt:
        bins, naive, trained = do_simulation(mpath(base_path_pt, p512))[:3]
        all_n.append(naive)
        t_512.append(trained)
    t_512 = np.vstack(t_512)
    all_n = np.vstack(all_n)
    fig, ax = pl.subplots()
    sns.tsplot(t_512, np.rad2deg(bins), n_boot=1000, ax=ax, color="C1", condition="Trained")
    sns.tsplot(all_n, np.rad2deg(bins), n_boot=1000, ax=ax, color="k", condition="Naive")
    ax.plot([0, 0], ax.get_ylim(), 'k--')
    ax.set_ylim(0)
    ax.legend()
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Angle to light source")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "Phototaxis_navigation.pdf", type="pdf")

    # Panel 4 - PCA space comparison of zfish gradient and phototaxis responses
    all_cells = np.hstack((a.trial_average(all_cells_zf, 3), a.trial_average(all_cells_pt, 3))).T
    max_vals = np.max(all_cells, 1, keepdims=True)
    max_vals[max_vals == 0] = 1  # these cells do not show any response
    all_cells /= max_vals
    species_id = np.zeros(all_cells.shape[0])
    species_id[all_cells_zf.shape[1]:] = 1
    pca = PCA(4)
    pca.fit(all_cells)
    coords = pca.transform(all_cells)
    for i in range(pca.n_components):
        plot_pc(i, coords, species_id, pca.explained_variance_, "zf_pt")

    # Panel 6: Gradient navigation performance of C elegans model
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
    fig.savefig(save_folder+"ce_gradient_distribution.pdf", type="pdf")

    # Panel 7: Temperature step responses of putative AFD/AWC neurons
    afd_like = 1
    awc_like = 7
    step_min = 23
    step_max = 27
    temp_step = np.zeros(temperature.size // 3)
    temp_step[:temp_step.size//5] = step_min
    temp_step[temp_step.size*4//5:] = step_max
    ramp = temp_step[temp_step.size//5:temp_step.size*4//5]
    ramp = np.arange(ramp.size)/ramp.size*(step_max-step_min) + step_min
    temp_step[temp_step.size//5:temp_step.size*4//5] = ramp
    cells_ce_step = []
    for i, p in enumerate(paths_512_ce):
        cell_res, ids = ana_ce.temperature_activity(mpath(base_path_ce, p), temp_step, i)
        cells_ce_step.append(cell_res)
    cells_ce_step = np.hstack(cells_ce_step)
    for i in range(cells_ce_step.shape[1]):
        cells_ce_step[:, i] = convolve(cells_ce_step[:, i], kernel, method='full')[:cells_ce_step.shape[0]]
    afd_data = cells_ce_step[:, clust_ids_ce == afd_like].T
    awc_data = cells_ce_step[:, clust_ids_ce == awc_like].T
    trial_time = np.arange(cells_ce_step.shape[0]) / GlobalDefs.frame_rate
    fig, ax = pl.subplots()
    sns.tsplot(afd_data, trial_time, ax=ax, color="C3")
    sns.tsplot(awc_data, trial_time, ax=ax, color="C1")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Activation [AU]")
    ax.set_xticks([0, 30, 60, 90, 120, 150])
    sns.despine(fig, ax)
    fig.savefig(save_folder + "ce_step_responses.pdf", type="pdf")
    fig, ax = pl.subplots()
    ax.plot(trial_time, temp_step, 'k')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [C]")
    ax.set_xticks([0, 30, 60, 90, 120, 150])
    sns.despine(fig, ax)
    fig.savefig(save_folder + "step_stimulus.pdf", type="pdf")

    # Panel 8: PCA space comparison of zfish and c elegans gradient responses
    all_cells = np.hstack((a.trial_average(all_cells_zf, 3), a.trial_average(all_cells_ce, 3))).T
    max_vals = np.max(all_cells, 1, keepdims=True)
    max_vals[max_vals == 0] = 1  # these cells do not show any response
    all_cells /= max_vals
    species_id = np.zeros(all_cells.shape[0])
    species_id[all_cells_zf.shape[1]:] = 1
    pca = PCA(4)
    pca.fit(all_cells)
    coords = pca.transform(all_cells)
    for i in range(pca.n_components):
        plot_pc(i, coords, species_id, pca.explained_variance_, "zf_ce")

    # Panel 9: Robustness of C elegans network to random deletions
    percent_to_remove = [25, 50, 75, 85, 90, 95, 97, 99]
    rem_d = {"state": [], "values": [], "species": []}
    # store the random removal drop-lists to disk so that we can quickly re-make this panel from
    # stored simulation results - as these depend on the drop-list they would never be loaded
    # if drop-lists are randomized every time
    dlist_file = h5py.File("drop_lists.hdf5")
    for i, p in enumerate(paths_512_ce):
        mp = mpath(base_path_ce, p)
        pos = ana_ce.run_simulation(mp, "r", "naive")
        rem_d["values"].append(a.preferred_fraction(pos, "r", 1.0))
        rem_d["state"].append("naive")
        rem_d["species"].append("C elegans")
        pos = ana_ce.run_simulation(mp, "r", "trained")
        rem_d["values"].append(a.preferred_fraction(pos, "r", 1.0))
        rem_d["state"].append("trained")
        rem_d["species"].append("C elegans")
        for ptr in percent_to_remove:
            file_key = mp + "_{0}".format(ptr)
            rem_d["state"].append("{0} %".format(ptr))
            rem_d["species"].append("C elegans")
            rand_clusts = np.zeros(all_ids_ce.shape[1])
            nw_units = rand_clusts.size // (len(paths_512_ce)*2)
            if file_key in dlist_file:
                dlist = pickle.loads(np.array(dlist_file[file_key]))
            else:
                for j in range(len(paths_512_ce)*2):
                    rand_clusts[j * nw_units:j * nw_units + int(nw_units * ptr / 100)] = 1
                dlist = a.create_det_drop_list(i, rand_clusts, all_ids_ce, [1], True)
                dlist_file.create_dataset(file_key, data=np.void(pickle.dumps(dlist, pickle.HIGHEST_PROTOCOL)))
            pos = ana_ce.run_simulation(mp, "r", "trained", drop_list=dlist)
            rem_d["values"].append(a.preferred_fraction(pos, "r", 1.0))
    dlist_file.close()
    rem_d = DataFrame(rem_d)
    fig, ax = pl.subplots()
    sns.pointplot("state", "values", "species", rem_d, ci=68, ax=ax)
    ax.set_ylabel("Fraction within +/- 1C")
    ax.set_xlabel("")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "ce_network_stability.pdf", type="pdf")

    # Panel 10: Full distribution of example type removals in C.elegans
    pal = sns.color_palette()  # the default matplotlib color cycle
    plot_cols_ce = {0: (0.6, 0.6, 0.6), 1: pal[3], 2: (0.6, 0.6, 0.6), 3: (0.6, 0.6, 0.6), 4: (0.6, 0.6, 0.6),
                    5: (0.6, 0.6, 0.6), 6: (0.6, 0.6, 0.6), 7: pal[1], "naive": (0.0, 0.0, 0.0),
                    "trained": (0.9, 0.9, 0.9), -1: (0.6, 0.6, 0.6)}
    # for worm-like clusters - their indices
    afd_like = 1
    awc_like = 7
    trained = np.empty((len(paths_512_ce), centers.size))
    afd_rem = np.empty_like(trained)
    awc_rem = np.empty_like(trained)
    for i, p in enumerate(paths_512_ce):
        mp = mpath(base_path_ce, p)
        pos_t = ana_ce.run_simulation(mp, "r", "trained")
        trained[i, :] = a.bin_simulation(pos_t, bns, "r")
        dlist = a.create_det_drop_list(i, clust_ids_ce, all_ids_ce, [afd_like])
        pos = ana_ce.run_simulation(mp, "r", "trained", drop_list=dlist)
        afd_rem[i, :] = a.bin_simulation(pos, bns, "r")
        dlist = a.create_det_drop_list(i, clust_ids_ce, all_ids_ce, [awc_like])
        pos = ana_ce.run_simulation(mp, "r", "trained", drop_list=dlist)
        awc_rem[i, :] = a.bin_simulation(pos, bns, "r")
    fig, ax = pl.subplots()
    sns.tsplot(trained, centers, n_boot=1000, condition="Trained", color="k")
    sns.tsplot(afd_rem, centers, n_boot=1000, condition="AFD", color=plot_cols_ce[afd_like])
    sns.tsplot(awc_rem, centers, n_boot=1000, condition="AWC/AIY", color=plot_cols_ce[awc_like])
    ax.plot([GlobalDefs.tPreferred, GlobalDefs.tPreferred], [0, 0.075], 'k--', lw=0.25)
    ax.legend()
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel("Proportion")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "ce_rem_gradient_distribution.pdf", type="pdf")

    # Panel 11: Aggregated type removals in C. elegans
    rem_dict = {i: [] for i in range(8)}
    rem_dict[-1] = []
    rem_dict["naive"] = []
    rem_dict["trained"] = []
    plot_order = ["naive", "trained", 1, 7, 0, 2, 3, 4, 5, 6, -1]
    plot_cols = [plot_cols_ce[k] for k in plot_order]
    for i, p in enumerate(paths_512_ce):
        mp = mpath(base_path_ce, p)
        rem_dict["naive"].append(a.preferred_fraction(ana_ce.run_simulation(mp, "r", "naive"), "r"))
        rem_dict["trained"].append(a.preferred_fraction(ana_ce.run_simulation(mp, "r", "trained"), "r"))
        for cc in range(8):
            dlist = a.create_det_drop_list(i, clust_ids_ce, all_ids_ce, [cc])
            rem_dict[cc].append(a.preferred_fraction(ana_ce.run_simulation(mp, "r", "trained", drop_list=dlist), "r"))
        dlist = a.create_det_drop_list(i, clust_ids_ce, all_ids_ce, [-1])
        rem_dict[-1].append(a.preferred_fraction(ana_ce.run_simulation(mp, "r", "trained", drop_list=dlist), "r"))
    rem_ce = DataFrame(rem_dict)
    fig, ax = pl.subplots()
    sns.barplot(data=rem_ce, order=plot_order, palette=plot_cols, ci=68)
    sns.despine(fig, ax)
    ax.set_ylabel("Fraction within +/- 1C")
    ax.set_yticks([0, 0.25, 0.5, 0.75])
    fig.savefig(save_folder + "ce_type_ablations.pdf", type="pdf")

    # Panel 12: Step-responses of strong phenotype types
    type0_data = cells_ce_step[:, clust_ids_ce == 0].T
    type2_data = cells_ce_step[:, clust_ids_ce == 2].T
    fig, ax = pl.subplots()
    sns.tsplot(type0_data, trial_time, ax=ax, color="C0")
    sns.tsplot(type2_data, trial_time, ax=ax, color="C2")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Activation [AU]")
    ax.set_xticks([0, 30, 60, 90, 120, 150])
    sns.despine(fig, ax)
    fig.savefig(save_folder + "ce_phenotype_step_responses.pdf", type="pdf")
