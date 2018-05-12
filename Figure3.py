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
from pandas import DataFrame
import pickle
from data_stores import SimulationStore, ActivityStore
from multiprocessing import Pool


# file definitions
base_path_zf = "./model_data/Adam_1e-4/sepInput_mixTrain/"
paths_512_zf = [f + '/' for f in os.listdir(base_path_zf) if "_3m512_" in f]

_mpool = None


def mpath(base, path):
    return base + path[:-1]  # need to remove trailing slash


def shutdown_pool():
    global _mpool
    if _mpool is not None:
        _mpool.close()
        _mpool.join()
        _mpool = None


def get_pool():
    global _mpool
    if _mpool is None:
        cp_count = os.cpu_count()
        if cp_count is None:
            cp_count = 2
        _mpool = Pool(cp_count // 2)
    return _mpool


def get_best_fit(activities, regressors):
    """
    For each cell in activities returns the index of the best fitting regressor >0.6 or -1 if none found
    Correlations are performed after calcium convolution
    :param activities: n_timepoints x m_cells matrix of cell activations
    :param regressors: n_timepoints x k_regressors matrix of regressors
    :return: m long vector of best fit regressor indices
    """
    # build calcium convolution kernel
    tau_on = 1.4  # seconds
    tau_on *= GlobalDefs.frame_rate  # in frames
    tau_off = 2  # seconds
    tau_off *= GlobalDefs.frame_rate  # in frames
    kframes = np.arange(10 * GlobalDefs.frame_rate)  # 10 s long kernel
    kernel = 2**(-kframes / tau_off) * (1 - 2**(-kframes / tau_on))
    kernel = kernel / kernel.sum()
    fit_vec = np.zeros(activities.shape[1])
    for i in range(activities.shape[1]):
        corrs = np.empty(regressors.shape[1])
        for j in range(corrs.size):
            corrs[j] = np.corrcoef(c.ca_convolve(activities[:, i], 0, 0, kernel),
                                   c.ca_convolve(regressors[:, j], 0, 0, kernel))[0, 1]
        corrs[np.isnan(corrs)] = 0
        if corrs.max() < 0.6:
            fit_vec[i] = -1
        else:
            fit_vec[i] = np.argmax(corrs)
    return fit_vec


def get_cluster_assignments(mt: MoTypes, model_dir: str, regressors, t_stimulus, std, droplist):
    """
    Creates a dictionary of cluster assignments for cells in t and m branch of a model
    :param mt: The model organism to use
    :param model_dir: The folder of the model checkpoint
    :param regressors: The cluster regressors
    :param t_stimulus: The temperature stimulus to use
    :param std: The standardizations
    :param droplist: Unit drop list
    :return: Dictionary with 't' and 'm' unit responses to stimulus
    """
    md = c.ModelData(model_dir)
    ml = mt.network_model()
    ml.load(md.ModelDefinition, md.LastCheckpoint)
    # prepend lead-in to stimulus
    lead_in = np.full(ml.input_dims[2] - 1, np.mean(t_stimulus[:10]))
    temp = np.r_[lead_in, t_stimulus]
    act_dict = ml.unit_stimulus_responses(temp, None, None, std, droplist)
    mpool = get_pool()
    ares = {k: [mpool.apply_async(get_best_fit, (ad, regressors)) for ad in act_dict[k]] for k in ['t', 'm']}
    retval = {k: np.vstack([ar.get() for ar in ares[k]]) for k in ares}
    return retval


def test_loss_zf_retrain(path):
    fname = base_path_zf + path + "fl_nontbranch_retrain/losses.hdf5"
    lossfile = h5py.File(fname, "r")
    rank_errors_non_t = np.array(lossfile["test_rank_errors"])
    fname = base_path_zf + path + "fl_tbranch_retrain/losses.hdf5"
    lossfile = h5py.File(fname, "r")
    rank_errors_t = np.array(lossfile["test_rank_errors"])
    timepoints = np.array(lossfile["test_eval"])
    return timepoints, rank_errors_t, rank_errors_non_t


if __name__ == "__main__":
    save_folder = "./DataFigures/Figure3/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42

    std_zf = c.GradientData.load_standards("gd_training_data.hdf5")
    ana_zf = a.Analyzer(MoTypes(False), std_zf, "sim_store.hdf5", "activity_store.hdf5")

    # for fish network clusters - their indices matched to plot colors to match Figure 2
    pal = sns.color_palette()  # the default matplotlib color cycle
    plot_cols_zf = {0: (0.6, 0.6, 0.6), 1: pal[2], 2: (102/255, 45/255, 145/255), 3: pal[0], 4: pal[3], 5: pal[1],
                    6: (0.6, 0.6, 0.6), 7: (0.6, 0.6, 0.6), "naive": (0.0, 0.0, 0.0), "trained": (0.9, 0.9, 0.9)}

    # load activity clusters from file
    clfile = h5py.File("cluster_info.hdf5", "r")
    clust_ids_zf = np.array(clfile["clust_ids"])
    clfile.close()

    # load and interpolate temperature stimulus
    dfile = h5py.File("stimFile.hdf5", 'r')
    tsin = np.array(dfile['sine_L_H_temp'])
    x = np.arange(tsin.size)  # stored at 20 Hz !
    xinterp = np.linspace(0, tsin.size, tsin.size * GlobalDefs.frame_rate // 20)
    temperature = np.interp(xinterp, x, tsin)
    dfile.close()

    # get activity data
    all_ids_zf = []
    all_cells_zf = []
    for i, p in enumerate(paths_512_zf):
        cell_res, ids = ana_zf.temperature_activity(mpath(base_path_zf, p), temperature, i)
        all_ids_zf.append(ids)
        all_cells_zf.append(cell_res)
    all_ids_zf = np.hstack(all_ids_zf)
    all_cells_zf = np.hstack(all_cells_zf)

    # panel 1: Robustness of zfish network to random removals - gradient performance vs. percent removed
    percent_to_remove = [25, 50, 75, 85, 90, 95, 97, 99]
    rem_d = {"state": [], "values": [], "species": []}
    # store the random removal drop-lists to disk so that we can quickly re-make this panel from
    # stored simulation results - as these depend on the drop-list they would never be loaded
    # if drop-lists are randomized every time
    dlist_file = h5py.File("drop_lists.hdf5")
    for i, p in enumerate(paths_512_zf):
        mp = mpath(base_path_zf, p)
        pos = ana_zf.run_simulation(mp, "r", "naive")
        rem_d["values"].append(a.preferred_fraction(pos, "r", 1.0))
        rem_d["state"].append("naive")
        rem_d["species"].append("zebrafish")
        pos = ana_zf.run_simulation(mp, "r", "bfevolve")
        rem_d["values"].append(a.preferred_fraction(pos, "r", 1.0))
        rem_d["state"].append("trained")
        rem_d["species"].append("zebrafish")
        for ptr in percent_to_remove:
            file_key = mp + "_{0}".format(ptr)
            rem_d["state"].append("{0} %".format(ptr))
            rem_d["species"].append("zebrafish")
            rand_clusts = np.zeros(all_ids_zf.shape[1])
            nw_units = rand_clusts.size // (len(paths_512_zf)*2)  # assume 2 layers in t branch
            if file_key in dlist_file:
                dlist = pickle.loads(np.array(dlist_file[file_key]))
            else:
                # loop through each invidual layer removing desired number of units (since we never shuffle
                # between layers)
                for j in range(len(paths_512_zf)*2):
                    rand_clusts[j * nw_units:j * nw_units + int(nw_units * ptr / 100)] = 1
                dlist = a.create_det_drop_list(i, rand_clusts, all_ids_zf, [1], True)
                dlist_file.create_dataset(file_key, data=np.void(pickle.dumps(dlist, pickle.HIGHEST_PROTOCOL)))
            pos = ana_zf.run_simulation(mp, "r", "bfevolve", drop_list=dlist)
            rem_d["values"].append(a.preferred_fraction(pos, "r", 1.0))
    dlist_file.close()
    rem_d = DataFrame(rem_d)
    fig, ax = pl.subplots()
    sns.pointplot("state", "values", "species", rem_d, ci=68, ax=ax)
    ax.set_ylabel("Fraction within +/- 1C")
    ax.set_xlabel("")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "zf_network_stability.pdf", type="pdf")

    # panel 2: Full distribution of example type removals in zebrafish
    # for fish-like clusters - their indices
    fast_on_like = 4
    slow_on_like = 5
    fast_off_like = 1
    slow_off_like = 3
    int_off = 2
    bns = np.linspace(0, GlobalDefs.circle_sim_params["radius"], 100)
    centers = a.temp_convert(bns[:-1] + np.diff(bns), "r")
    evolved = np.empty((len(paths_512_zf), centers.size))
    f_on_rem = np.empty_like(evolved)
    s_on_rem = np.empty_like(evolved)
    f_off_rem = np.empty_like(evolved)
    int_off_rem = np.empty_like(evolved)
    for i, p in enumerate(paths_512_zf):
        mp = mpath(base_path_zf, p)
        pos_e = ana_zf.run_simulation(mp, "r", "bfevolve")
        evolved[i, :] = a.bin_simulation(pos_e, bns, "r")
        dlist = a.create_det_drop_list(i, clust_ids_zf, all_ids_zf, [fast_on_like])
        pos = ana_zf.run_simulation(mp, "r", "bfevolve", drop_list=dlist)
        f_on_rem[i, :] = a.bin_simulation(pos, bns, "r")
        dlist = a.create_det_drop_list(i, clust_ids_zf, all_ids_zf, [slow_on_like])
        pos = ana_zf.run_simulation(mp, "r", "bfevolve", drop_list=dlist)
        s_on_rem[i, :] = a.bin_simulation(pos, bns, "r")
        dlist = a.create_det_drop_list(i, clust_ids_zf, all_ids_zf, [fast_off_like])
        pos = ana_zf.run_simulation(mp, "r", "bfevolve", drop_list=dlist)
        f_off_rem[i, :] = a.bin_simulation(pos, bns, "r")
        dlist = a.create_det_drop_list(i, clust_ids_zf, all_ids_zf, [int_off])
        pos = ana_zf.run_simulation(mp, "r", "bfevolve", drop_list=dlist)
        int_off_rem[i, :] = a.bin_simulation(pos, bns, "r")
    fig, ax = pl.subplots()
    sns.tsplot(evolved, centers, n_boot=1000, condition="trained", color="k")
    sns.tsplot(f_on_rem, centers, n_boot=1000, condition="Fast ON", color=plot_cols_zf[fast_on_like])
    sns.tsplot(s_on_rem, centers, n_boot=1000, condition="Slow ON", color=plot_cols_zf[slow_on_like])
    sns.tsplot(f_off_rem, centers, n_boot=1000, condition="Fast OFF", color=plot_cols_zf[fast_off_like])
    sns.tsplot(int_off_rem, centers, n_boot=1000, condition="Integr. OFF", color=plot_cols_zf[int_off])
    ax.plot([GlobalDefs.tPreferred, GlobalDefs.tPreferred], [0, 0.05], 'k--', lw=0.25)
    ax.legend()
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel("Proportion")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "zf_rem_gradient_distribution.pdf", type="pdf")

    # panel 3: Aggregated type removals in zebrafish
    rem_dict = {i: [] for i in range(8)}
    rem_dict["naive"] = []
    rem_dict["trained"] = []
    plot_order = ["naive", "trained", 4, 5, 1, 3, 2, 0, 6, 7]
    plot_cols = [plot_cols_zf[k] for k in plot_order]
    for i, p in enumerate(paths_512_zf):
        mp = mpath(base_path_zf, p)
        rem_dict["naive"].append(a.preferred_fraction(ana_zf.run_simulation(mp, "r", "naive"), "r"))
        rem_dict["trained"].append(a.preferred_fraction(ana_zf.run_simulation(mp, "r", "bfevolve"), "r"))
        for cc in range(8):
            dlist = a.create_det_drop_list(i, clust_ids_zf, all_ids_zf, [cc])
            rem_dict[cc].append(a.preferred_fraction(ana_zf.run_simulation(mp, "r", "bfevolve", drop_list=dlist), "r"))
    rem_zf = DataFrame(rem_dict)
    fig, ax = pl.subplots()
    sns.barplot(data=rem_zf, order=plot_order, palette=plot_cols, ci=68)
    sns.despine(fig, ax)
    ax.set_ylabel("Fraction within +/- 1C")
    ax.set_yticks([0, 0.25, 0.5])
    fig.savefig(save_folder + "zf_type_ablations.pdf", type="pdf")

    # for each removal and retrain compute gradient distributions
    trained = np.empty((len(paths_512_zf), centers.size))
    fl_ablated = np.empty_like(trained)
    fl_retrained_t = np.empty_like(trained)
    fl_retrained_nont = np.empty_like(trained)
    nfl_ablated = np.empty_like(trained)
    for i, p in enumerate(paths_512_zf):
        mp = mpath(base_path_zf, p)
        rt_path_nont = mp + "/fl_nontbranch_retrain"
        rt_path_t = mp + "/fl_tbranch_retrain"
        with SimulationStore(None, std_zf, MoTypes(False)) as sim_store:
            pos = sim_store.get_sim_pos(mp, 'r', "trained")
            trained[i, :] = a.bin_simulation(pos, bns, 'r')
            dlist = a.create_det_drop_list(i, clust_ids_zf, all_ids_zf, [1, 2, 3, 4, 5])
            pos = sim_store.get_sim_pos(mp, 'r', "trained", dlist)
            fl_ablated[i, :] = a.bin_simulation(pos, bns, 'r')
            pos = sim_store.get_sim_pos(rt_path_t, 'r', "trained", dlist)
            fl_retrained_t[i, :] = a.bin_simulation(pos, bns, 'r')
            pos = sim_store.get_sim_pos(rt_path_nont, 'r', "trained", dlist)
            fl_retrained_nont[i, :] = a.bin_simulation(pos, bns, 'r')
            dlist = a.create_det_drop_list(i, clust_ids_zf, all_ids_zf, [0, 6, 7])
            pos = sim_store.get_sim_pos(mp, 'r', "trained", dlist)
            nfl_ablated[i, :] = a.bin_simulation(pos, bns, 'r')

    # panel 4: Consequence of ablating all fish or non-fish types
    fig, ax = pl.subplots()
    sns.tsplot(trained, centers, n_boot=1000, condition="Trained", color="k")
    sns.tsplot(fl_ablated, centers, n_boot=1000, condition="Fish-like ablated", color="C1")
    sns.tsplot(nfl_ablated, centers, n_boot=1000, condition="Non-fish ablated", color="C3")
    ax.plot([GlobalDefs.tPreferred, GlobalDefs.tPreferred], [0, 0.03], 'k--', lw=0.25)
    ax.legend()
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel("Proportion")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "zf_fish_nonfish_ablation_distribution.pdf", type="pdf")

    # panel 5: Rank errors during re-training
    re_t_branch = []
    re_nont_branch = []
    test_times = None
    for p in paths_512_zf:
        test_times, t, non_t = test_loss_zf_retrain(p)
        re_t_branch.append(t)
        re_nont_branch.append(non_t)
    re_t_branch = np.vstack(re_t_branch)
    re_nont_branch = np.vstack(re_nont_branch)
    fig, ax = pl.subplots()
    sns.tsplot(re_t_branch, test_times, ci=68, color="C3", condition="Temperature branch only")
    sns.tsplot(re_nont_branch, test_times, ci=68, color="C0", condition="Mixed branch only")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "zf_fish_retrain_rank_errors.pdf", type="pdf")

    # panel 6: Retraining after ablating all fish types
    fig, ax = pl.subplots()
    sns.tsplot(fl_ablated, centers, n_boot=1000, condition="Ablated", color="k")
    sns.tsplot(trained, centers, n_boot=1000, condition="Trained", color='k')
    sns.tsplot(fl_retrained_t, centers, n_boot=1000, condition="Temperature part retrained", color="C1")
    sns.tsplot(fl_retrained_nont, centers, n_boot=1000, condition="Shared part retrained", color="C3")
    ax.plot([GlobalDefs.tPreferred, GlobalDefs.tPreferred], [0, 0.03], 'k--', lw=0.25)
    ax.legend()
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel("Proportion")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "zf_fish_retrained_distribution.pdf", type="pdf")

    # panel 7-8: Type fractions pre-ablation, post-ablation and after full re-training
    # make cluster regressors
    regs_zf = np.zeros((all_cells_zf.shape[0], np.unique(clust_ids_zf).size - 1))
    for cnum in np.unique(clust_ids_zf):
        if cnum == -1:
            continue
        regs_zf[:, cnum] = np.mean(all_cells_zf[:, clust_ids_zf == cnum], 1)
    type_fractions = {"cell type": [], "state": [], "fraction": [], "network_id": [], "branch": []}
    # get cell responses and populate our fractions
    for i, p in enumerate(paths_512_zf):
        mp = mpath(base_path_zf, p)
        rt_path_nont = mp + "/fl_nontbranch_retrain"
        rt_path_t = mp + "/fl_tbranch_retrain"
        with ActivityStore(None, std_zf, MoTypes(False)) as act_store:
            clusters_trained = get_cluster_assignments(MoTypes(False), mp, regs_zf, temperature, std_zf, None)
            dlist = a.create_det_drop_list(i, clust_ids_zf, all_ids_zf, [1, 2, 3, 4, 5])
            clusters_ablated = get_cluster_assignments(MoTypes(False), mp, regs_zf, temperature, std_zf, dlist)
            clusters_retr_t = get_cluster_assignments(MoTypes(False), rt_path_t, regs_zf, temperature, std_zf, dlist)
            clusters_retr_nont = get_cluster_assignments(MoTypes(False), rt_path_nont, regs_zf, temperature, std_zf,
                                                         dlist)
            for cnum in np.unique(clust_ids_zf):
                for br in ['t', 'm']:
                    if cnum == 4:
                        ctype = "Fast ON"
                    elif cnum == 5:
                        ctype = "Slow ON"
                    elif cnum == 1:
                        ctype = "Fast OFF"
                    elif cnum == 3:
                        ctype = "Slow OFF"
                    elif cnum == 2:
                        ctype = "Int. OFF"
                    elif cnum == -1:
                        ctype = "Unassigned"
                    else:
                        ctype = "Non fish"
                    type_fractions["cell type"] += [ctype]*4
                    type_fractions["branch"] += [br]*4
                    type_fractions["network_id"] += [i]*4
                    type_fractions["state"].append("Trained")
                    type_fractions["fraction"].append(np.sum(clusters_trained[br] == cnum)/clusters_trained[br].size)
                    type_fractions["state"].append("Ablated")
                    type_fractions["fraction"].append(np.sum(clusters_ablated[br] == cnum) / clusters_ablated[br].size)
                    type_fractions["state"].append("Retrained T")
                    type_fractions["fraction"].append(np.sum(clusters_retr_t[br] == cnum) / clusters_retr_t[br].size)
                    type_fractions["state"].append("Retrained M")
                    type_fractions["fraction"].append(np.sum(clusters_retr_nont[br] == cnum) /
                                                      clusters_retr_nont[br].size)
    df_typefrac = DataFrame(type_fractions)
    order = ["Fast ON", "Slow ON", "Fast OFF", "Slow OFF", "Int. OFF", "Non fish", "Unassigned"]
    # panel 9: type fractions after ablation and re-training in temperature branch
    fig, ax = pl.subplots()
    sns.barplot(x="cell type", y="fraction", hue="state", data=df_typefrac[df_typefrac["branch"] == 't'], ci=68,
                order=order)
    sns.despine(fig, ax)
    fig.savefig(save_folder + "zf_ablation_and_retrain_tbranch_tfracs.pdf", type="pdf")

    # panel 10: type fractions after ablation and re-training in mixed branch
    fig, ax = pl.subplots()
    sns.barplot(x="cell type", y="fraction", hue="state", data=df_typefrac[df_typefrac["branch"] == 'm'], ci=68,
                order=order)
    sns.despine(fig, ax)
    fig.savefig(save_folder + "zf_ablation_and_retrain_mbranch_tfracs.pdf", type="pdf")

    shutdown_pool()
