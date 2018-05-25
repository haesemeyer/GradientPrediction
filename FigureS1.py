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
from data_stores import SimulationStore
from Figure3 import mpath

# file definitions
base_path_zf = "./model_data/Adam_1e-4/sepInput_mixTrain/"
paths_512_zf = [f + '/' for f in os.listdir(base_path_zf) if "_3m512_" in f]


if __name__ == "__main__":
    save_folder = "./DataFigures/FigureS1/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42
    std_zf = c.GradientData.load_standards("gd_training_data.hdf5")

    # for each complete branch removal compute gradient distributions
    bns = np.linspace(0, GlobalDefs.circle_sim_params["radius"], 100)
    centers = a.temp_convert(bns[:-1] + np.diff(bns), "r")
    evolved = np.empty((len(paths_512_zf), centers.size))
    t_ablated = np.empty_like(evolved)
    s_ablated = np.empty_like(evolved)
    a_ablated = np.empty_like(evolved)
    dlist_all_t = {'t': [np.zeros(512), np.zeros(512)]}
    dlist_all_s = {'s': [np.zeros(512), np.zeros(512)]}
    dlist_all_a = {'a': [np.zeros(512), np.zeros(512)]}
    for i, p in enumerate(paths_512_zf):
        mp = mpath(base_path_zf, p)
        with SimulationStore("zf_full_branch_abl.hdf5", std_zf, MoTypes(False)) as sim_store:
            pos = sim_store.get_sim_pos(mp, 'r', "trained")
            evolved[i, :] = a.bin_simulation(pos, bns, 'r')
            pos = sim_store.get_sim_pos(mp, 'r', "trained", dlist_all_t)
            t_ablated[i, :] = a.bin_simulation(pos, bns, 'r')
            pos = sim_store.get_sim_pos(mp, 'r', "trained", dlist_all_s)
            s_ablated[i, :] = a.bin_simulation(pos, bns, 'r')
            pos = sim_store.get_sim_pos(mp, 'r', "trained", dlist_all_a)
            a_ablated[i, :] = a.bin_simulation(pos, bns, "r")

    # Panel: Full branch removal gradient navigation
    fig, ax = pl.subplots()
    sns.tsplot(evolved, centers, n_boot=1000, condition="trained", color="k")
    sns.tsplot(t_ablated, centers, n_boot=1000, condition="t branch", color="C3")
    sns.tsplot(s_ablated, centers, n_boot=1000, condition="s branch", color="C0")
    sns.tsplot(a_ablated, centers, n_boot=1000, condition="a branch", color="C2")
    ax.plot([GlobalDefs.tPreferred, GlobalDefs.tPreferred], [0, 0.03], 'k--', lw=0.25)
    ax.legend()
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel("Proportion")
    sns.despine(fig, ax)
    fig.savefig(save_folder + "zf_branch_rem_gradient_distribution.pdf", type="pdf")
