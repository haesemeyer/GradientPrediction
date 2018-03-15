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
from Figure1 import mpath
from mo_types import MoTypes
import core as c
import analysis as a
import h5py
from global_defs import GlobalDefs
from pandas import DataFrame


# file definitions
base_path = "./model_data/Adam_1e-4/sepInput_mixTrain/"

paths_512 = [f+'/' for f in os.listdir(base_path) if "_3m512_" in f]


if __name__ == "__main__":
    save_folder = "./DataFigures/Figure4/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42
    mo = MoTypes(False)
    std = c.GradientData.load_standards("gd_training_data.hdf5")
    ana_zf = a.Analyzer(mo, std, "sim_store.hdf5", "activity_store.hdf5")

    # for fish network clusters - their indices matched to plot colors to match Figure 2
    pal = sns.color_palette()  # the default matplotlib color cycle
    plot_colors = {0: (0.6, 0.6, 0.6), 1: pal[2], 2: (0.6, 0.6, 0.6), 3: pal[0], 4: pal[3], 5: pal[1],
                   6: (0.6, 0.6, 0.6), 7: (0.6, 0.6, 0.6), "naive": (0.0, 0.0, 0.0), "trained": (0.9, 0.9, 0.9)}

    # load activity clusters from file
    clfile = h5py.File("cluster_info.hdf5", "r")
    clust_ids = np.array(clfile["clust_ids"])
    clfile.close()

    # load and interpolate temperature stimulus
    dfile = h5py.File("stimFile.hdf5", 'r')
    tsin = np.array(dfile['sine_L_H_temp'])
    x = np.arange(tsin.size)  # stored at 20 Hz !
    xinterp = np.linspace(0, tsin.size, tsin.size * GlobalDefs.frame_rate // 20)
    temperature = np.interp(xinterp, x, tsin)
    dfile.close()

    # get activity data
    # all_cells = []
    all_ids = []
    for i, p in enumerate(paths_512):
        cell_res, ids = ana_zf.temperature_activity(mpath(p), temperature, i)
        # all_cells.append(cell_res)
        all_ids.append(ids)
    # all_cells = np.hstack(all_cells)
    all_ids = np.hstack(all_ids)

    # panel 2: Type removals in zebrafish
    rem_dict = {i: [] for i in range(8)}
    rem_dict["naive"] = []
    rem_dict["trained"] = []
    plot_order = ["naive", "trained", 4, 5, 1, 3, 0, 2, 6, 7]
    plot_cols = [plot_colors[k] for k in plot_order]
    for i, p in enumerate(paths_512):
        mp = mpath(p)
        rem_dict["naive"].append(a.preferred_fraction(ana_zf.run_simulation(mp, "r", "naive"), "r"))
        rem_dict["trained"].append(a.preferred_fraction(ana_zf.run_simulation(mp, "r", "bfevolve"), "r"))
        for cc in range(8):
            dlist = a.create_det_drop_list(i, clust_ids, all_ids, [cc])
            rem_dict[cc].append(a.preferred_fraction(ana_zf.run_simulation(mp, "r", "bfevolve", drop_list=dlist), "r"))
    rem_zf = DataFrame(rem_dict)
    fig, ax = pl.subplots()
    sns.barplot(data=rem_zf, order=plot_order, palette=plot_colors, ci=68)
    sns.despine(fig, ax)
    ax.set_ylabel("Fraction within +/- 1C")
    ax.set_yticks([0, 0.25, 0.5])
    fig.savefig(save_folder + "zf_type_ablations.pdf", type="pdf")
