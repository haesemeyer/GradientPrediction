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
                    5: (0.6, 0.6, 0.6), 6: (0.6, 0.6, 0.6), 7: (0.6, 0.6, 0.6), "naive": (0.0, 0.0, 0.0),
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

    # panel 2: Full distribution of example type removals in zebrafish

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
