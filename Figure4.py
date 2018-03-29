#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script for panels of Figure 4 (Ablations in C elegans and zebrafish networks)
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
from data_stores import SimulationStore


# file definitions
base_path_zf = "./model_data/Adam_1e-4/sepInput_mixTrain/"
paths_512_zf = [f + '/' for f in os.listdir(base_path_zf) if "_3m512_" in f]

base_path_ce = "./model_data/CE_Adam_1e-4/"
paths_512_ce = [f + '/' for f in os.listdir(base_path_ce) if "_3m512_" in f]


def mpath(base, path):
    return base + path[:-1]  # need to remove trailing slash


if __name__ == "__main__":
    save_folder = "./DataFigures/Figure4/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42

    std_zf = c.GradientData.load_standards("gd_training_data.hdf5")
    ana_zf = a.Analyzer(MoTypes(False), std_zf, "sim_store.hdf5", "activity_store.hdf5")
    std_ce = c.GradientData.load_standards("ce_gd_training_data.hdf5")
    ana_ce = a.Analyzer(MoTypes(True), std_ce, "ce_sim_store.hdf5", "ce_activity_store.hdf5")

    # for fish network clusters - their indices matched to plot colors to match Figure 2
    pal = sns.color_palette()  # the default matplotlib color cycle
    plot_cols_zf = {0: (0.6, 0.6, 0.6), 1: pal[2], 2: (0.6, 0.6, 0.6), 3: pal[0], 4: pal[3], 5: pal[1],
                    6: (0.6, 0.6, 0.6), 7: (0.6, 0.6, 0.6), "naive": (0.0, 0.0, 0.0), "trained": (0.9, 0.9, 0.9)}

    plot_cols_ce = {0: (0.6, 0.6, 0.6), 1: pal[3], 2: (0.6, 0.6, 0.6), 3: (0.6, 0.6, 0.6), 4: (0.6, 0.6, 0.6),
                    5: (0.6, 0.6, 0.6), 6: (0.6, 0.6, 0.6), 7: pal[1], "naive": (0.0, 0.0, 0.0),
                    "trained": (0.9, 0.9, 0.9)}

    # load activity clusters from file
    clfile = h5py.File("cluster_info.hdf5", "r")
    clust_ids_zf = np.array(clfile["clust_ids"])
    clfile.close()
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

    # get activity data
    all_ids_zf = []
    for i, p in enumerate(paths_512_zf):
        cell_res, ids = ana_zf.temperature_activity(mpath(base_path_zf, p), temperature, i)
        all_ids_zf.append(ids)
    all_ids_zf = np.hstack(all_ids_zf)
    all_ids_ce = []
    for i, p in enumerate(paths_512_ce):
        cell_res, ids = ana_ce.temperature_activity(mpath(base_path_ce, p), temperature, i)
        all_ids_ce.append(ids)
    all_ids_ce = np.hstack(all_ids_ce)

    # panel 1: Robustness of zfish and cele networks to random removals - gradient performance vs. percent removed
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
                print("Desired: {0}, actual 1: {1}, actual 2: {2}".format(ptr, 100*dlist['t'][0].sum()/512, 100*dlist['t'][1].sum()/512))
                dlist_file.create_dataset(file_key, data=np.void(pickle.dumps(dlist, pickle.HIGHEST_PROTOCOL)))
            pos = ana_zf.run_simulation(mp, "r", "bfevolve", drop_list=dlist)
            rem_d["values"].append(a.preferred_fraction(pos, "r", 1.0))
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
    fig.savefig(save_folder + "network_stability.pdf", type="pdf")

    # panel 2: Full distribution of example type removals in zebrafish
    # for fish-like clusters - their indices
    fast_on_like = 4
    slow_on_like = 5
    fast_off_like = 1
    slow_off_like = 3
    bns = np.linspace(0, GlobalDefs.circle_sim_params["radius"], 100)
    centers = a.temp_convert(bns[:-1] + np.diff(bns), "r")
    evolved = np.empty((len(paths_512_zf), centers.size))
    f_on_rem = np.empty_like(evolved)
    s_on_rem = np.empty_like(evolved)
    f_off_rem = np.empty_like(evolved)
    s_off_rem = np.empty_like(evolved)
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
        dlist = a.create_det_drop_list(i, clust_ids_zf, all_ids_zf, [slow_off_like])
        pos = ana_zf.run_simulation(mp, "r", "bfevolve", drop_list=dlist)
        s_off_rem[i, :] = a.bin_simulation(pos, bns, "r")
    fig, ax = pl.subplots()
    sns.tsplot(evolved, centers, n_boot=1000, condition="trained", color="k")
    sns.tsplot(f_on_rem, centers, n_boot=1000, condition="Fast ON", color=plot_cols_zf[fast_on_like])
    sns.tsplot(s_on_rem, centers, n_boot=1000, condition="Slow ON", color=plot_cols_zf[slow_on_like])
    sns.tsplot(f_off_rem, centers, n_boot=1000, condition="Fast OFF", color=plot_cols_zf[fast_off_like])
    sns.tsplot(s_off_rem, centers, n_boot=1000, condition="Slow OFF", color=plot_cols_zf[slow_off_like])
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
    plot_order = ["naive", "trained", 4, 5, 1, 3, 0, 2, 6, 7]
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

    # panel 4: Full distribution of example type removals in C. elegans
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

    # panel 5: Aggregated type removals in C. elegans
    rem_dict = {i: [] for i in range(8)}
    rem_dict["naive"] = []
    rem_dict["trained"] = []
    plot_order = ["naive", "trained", 1, 0, 2, 3, 4, 5, 6, 7]
    plot_cols = [plot_cols_ce[k] for k in plot_order]
    for i, p in enumerate(paths_512_ce):
        mp = mpath(base_path_ce, p)
        rem_dict["naive"].append(a.preferred_fraction(ana_ce.run_simulation(mp, "r", "naive"), "r"))
        rem_dict["trained"].append(a.preferred_fraction(ana_ce.run_simulation(mp, "r", "trained"), "r"))
        for cc in range(8):
            dlist = a.create_det_drop_list(i, clust_ids_ce, all_ids_ce, [cc])
            rem_dict[cc].append(a.preferred_fraction(ana_ce.run_simulation(mp, "r", "trained", drop_list=dlist), "r"))
    rem_ce = DataFrame(rem_dict)
    fig, ax = pl.subplots()
    sns.barplot(data=rem_ce, order=plot_order, palette=plot_cols, ci=68)
    sns.despine(fig, ax)
    ax.set_ylabel("Fraction within +/- 1C")
    ax.set_yticks([0, 0.25, 0.5, 0.75])
    fig.savefig(save_folder + "ce_type_ablations.pdf", type="pdf")

    # panel 6: Gradient distribution after fish-like ablations and re-training
    trained = np.empty((len(paths_512_zf), centers.size))
    fl_ablated = np.empty_like(trained)
    fl_retrained = np.empty_like(trained)
    for i, p in enumerate(paths_512_zf):
        mp = mpath(base_path_zf, p)
        rt_path = mp + "/fl_retrain"
        with SimulationStore(None, std_zf, MoTypes(False)) as sim_store:
            pos = sim_store.get_sim_pos(mp, 'r', "trained")
            trained[i, :] = a.bin_simulation(pos, bns, 'r')
            dlist = a.create_det_drop_list(i, clust_ids_zf, all_ids_zf, [1, 2, 3, 4, 5])
            pos = sim_store.get_sim_pos(mp, 'r', "trained", dlist)
            fl_ablated[i, :] = a.bin_simulation(pos, bns, 'r')
            pos = sim_store.get_sim_pos(rt_path, 'r', "trained", dlist)
            fl_retrained[i, :] = a.bin_simulation(pos, bns, 'r')
    fig, ax = pl.subplots()
    sns.tsplot(trained, centers, n_boot=1000, condition="Trained", color="k")
    sns.tsplot(fl_ablated, centers, n_boot=1000, condition="Ablated", color="C1")
    sns.tsplot(fl_retrained, centers, n_boot=1000, condition="Retrained", color="C3")
    ax.plot([GlobalDefs.tPreferred, GlobalDefs.tPreferred], [0, 0.03], 'k--', lw=0.25)
    ax.legend()
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel("Proportion")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "zf_ablation_and_retrain_distribution.pdf", type="pdf")
