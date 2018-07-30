#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script for panels of Figure S5 (Comparison with and structure of C elegans network)
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
from mo_types import MoTypes
from Figure4 import mpath
from scipy.signal import convolve
from pandas import DataFrame
import pickle


# file definitions
base_path_zf = "./model_data/Adam_1e-4/sepInput_mixTrain/"
paths_512_zf = [f + '/' for f in os.listdir(base_path_zf) if "_3m512_" in f]

base_path_ce = "./model_data/CE_Adam_1e-4/"
paths_512_ce = [f + '/' for f in os.listdir(base_path_ce) if "_3m512_" in f]


if __name__ == "__main__":
    save_folder = "./DataFigures/FigureS5/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42

    std_zf = c.GradientData.load_standards("gd_training_data.hdf5")
    ana_zf = a.Analyzer(MoTypes(False), std_zf, "sim_store.hdf5", "activity_store.hdf5")
    std_ce = c.GradientData.load_standards("ce_gd_training_data.hdf5")
    ana_ce = a.Analyzer(MoTypes(True), std_ce, "ce_sim_store.hdf5", "ce_activity_store.hdf5")

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
    trial_time = np.arange(all_cells_zf.shape[0] // 3) / GlobalDefs.frame_rate

    # plot colors
    pal = sns.color_palette()  # the default matplotlib color cycle
    plot_cols_ce = {0: pal[0], 1: pal[3], 2: pal[2], 3: pal[4], 4: pal[5],
                    5: (0.6, 0.6, 0.6), 6: pal[6], 7: pal[1], -1: (0.6, 0.6, 0.6)}
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

    # panel - all cluster activities, sorted into ON and OFF types
    n_regs = np.unique(clust_ids_ce).size - 1
    cluster_acts = np.zeros((cells_ce_step.shape[0], n_regs))
    for i in range(n_regs):
        cluster_acts[:, i] = np.mean(cells_ce_step[:, clust_ids_ce == i], 1)
    on_count = 0
    off_count = 0
    fig, (axes_on, axes_off) = pl.subplots(ncols=2, nrows=2, sharey=True, sharex=True)
    time = np.arange(cluster_acts.shape[0]) / GlobalDefs.frame_rate
    for i in range(n_regs):
        act = cluster_acts[:, i]
        if np.corrcoef(act, temp_step[:act.size])[0, 1] < 0:
            ax_off = axes_off[0] if off_count < 2 else axes_off[1]
            ax_off.plot(time, cluster_acts[:, i], color=plot_cols_ce[i])
            off_count += 1
        else:
            ax_on = axes_on[0] if on_count < 2 else axes_on[1]
            ax_on.plot(time, cluster_acts[:, i], color=plot_cols_ce[i])
            on_count += 1
    axes_off[0].set_xticks([0, 30, 60, 90, 120, 150])
    axes_off[1].set_xticks([0, 30, 60, 90, 120, 150])
    axes_off[0].set_xlabel("Time [s]")
    axes_off[1].set_xlabel("Time [s]")
    axes_on[0].set_ylabel("Cluster average activation")
    axes_off[0].set_ylabel("Cluster average activation")
    sns.despine()
    fig.tight_layout()
    fig.savefig(save_folder + "ce_all_cluster_averages.pdf", type="pdf")

    # panel - average type counts in temperature branch for each cluster
    cl_type_d = {"Fraction": [], "net_id": [], "Cluster ID": [], "Layer": []}
    for i in range(len(paths_512_zf)):
        for j in range(-1, n_regs):
            for k in range(2):
                lay_clust_ids = clust_ids_ce[np.logical_and(all_ids_ce[0, :] == i, all_ids_ce[1, :] == k)]
                cl_type_d["Fraction"].append(np.sum(lay_clust_ids == j) / 512)
                cl_type_d["net_id"].append(i)
                cl_type_d["Cluster ID"].append(j)
                cl_type_d["Layer"].append(k)
    cl_type_df = DataFrame(cl_type_d)
    fig, (ax_0, ax_1) = pl.subplots(nrows=2, sharex=True)
    sns.barplot("Cluster ID", "Fraction", data=cl_type_df[cl_type_df["Layer"] == 0], order=list(range(n_regs)) + [-1],
                ci=68, ax=ax_0, palette=plot_cols_ce)
    sns.barplot("Cluster ID", "Fraction", data=cl_type_df[cl_type_df["Layer"] == 1], order=list(range(n_regs)) + [-1],
                ci=68, ax=ax_1, palette=plot_cols_ce)
    ax_0.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax_1.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    sns.despine(fig)
    fig.savefig(save_folder + "ce_all_cluster_counts.pdf", type="pdf")

    # panel - input connectivity into second layer of t branch
    conn_mat = np.zeros((8, 8, len(paths_512_ce)))
    for i, p in enumerate(paths_512_ce):
        model_cids = clust_ids_ce[all_ids_ce[0, :] == i]
        layers_ids = all_ids_ce[1, :][all_ids_ce[0, :] == i]
        l_0_mask = np.full(8, False)
        ix = model_cids[layers_ids == 0]
        ix = ix[ix != -1]
        l_0_mask[ix] = True
        l_1_mask = np.full(8, False)
        ix = model_cids[layers_ids == 1]
        ix = ix[ix != -1]
        l_1_mask[ix] = True
        m_path = mpath(base_path_ce, p)
        mdata = c.ModelData(m_path)
        gpn = MoTypes(True).network_model()
        gpn.load(mdata.ModelDefinition, mdata.LastCheckpoint)
        input_result = gpn.parse_layer_input_by_cluster('t', 1, model_cids[layers_ids == 0],
                                                        model_cids[layers_ids == 1])
        for k, l0 in enumerate(np.arange(8)[l_0_mask]):
            for l, l1 in enumerate(np.arange(8)[l_1_mask]):
                conn_mat[l0, l1, i] = input_result[k, l]
    # reordered version of conn_mat based on known types
    cm_order = [1, 7, 0, 2, 3, 4, 5, 6]
    cm_reordered = conn_mat[:, cm_order, :]
    cm_reordered = cm_reordered[cm_order, :, :]
    m = np.mean(cm_reordered, 2)
    s = np.std(cm_reordered, 2)
    cross_0 = np.sign((m+s) * (m-s)) <= 0
    m[cross_0] = 0
    s[cross_0] = 0
    fig, axes = pl.subplots(nrows=4, sharex=True, sharey=True)
    for i in range(4):
        axes[i].bar(np.arange(8), m[:, i], width=[.8]*4+[.3]*4)
        axes[i].errorbar(np.arange(8), m[:, i], s[:, i], color='k', fmt='none')
    axes[-1].set_xticks(np.arange(8))
    axes[-1].set_xticklabels(["AFD", "AWC/AIY", "0", "2", "3", "4", "5", "6"])
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(save_folder + "ce_avg_connectivity_weights.pdf", type="pdf")

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
            nw_units = rand_clusts.size // (len(paths_512_ce) * 2)
            if file_key in dlist_file:
                dlist = pickle.loads(np.array(dlist_file[file_key]))
            else:
                for j in range(len(paths_512_ce) * 2):
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

    # Panels: Plot best-correlated pairs of zebrafish and C. elegans neurons
    clust_corr_mat = np.zeros((np.unique(clust_ids_zf).size-1, np.unique(clust_ids_ce).size-1))
    regs_ce = {}
    for i in range(clust_corr_mat.shape[0]):
        reg_zf = np.mean(all_cells_zf[:, clust_ids_zf == i], 1)
        for j in range(clust_corr_mat.shape[1]):
            if j in regs_ce:
                clust_corr_mat[i, j] = np.corrcoef(reg_zf, regs_ce[j])[0, 1]**2
            else:
                r = np.mean(all_cells_ce[:, clust_ids_ce == j], 1)
                regs_ce[j] = r
                clust_corr_mat[i, j] = np.corrcoef(reg_zf, r)[0, 1]**2

    def norm(trace):
        n = trace - trace.min()
        return n / n.max()

    ccm_copy = clust_corr_mat.copy()
    for i in range(np.max(ccm_copy.shape)):
        ix = np.argmax(ccm_copy)
        rw, cl = np.unravel_index(ix, ccm_copy.shape)
        ccm_copy[rw, :] = 0
        ccm_copy[:, cl] = 0
        fig, ax = pl.subplots()
        ax.plot(trial_time, norm(np.mean(a.trial_average(all_cells_zf[:, clust_ids_zf == rw], 3), 1)), color='k')
        ax.plot(trial_time, norm(np.mean(a.trial_average(all_cells_ce[:, clust_ids_ce == cl], 3), 1)), color='C1')
        ax.set_ylabel("Normalized activation")
        ax.set_xlabel("Time [s]")
        ax.set_title("Zf {0} vs Ce {1}. R^2 = {2}".format(rw, cl, np.round(clust_corr_mat[rw, cl], 2)))
        ax.set_xticks([0, 30, 60, 90, 120, 150])
        sns.despine(fig, ax)
        fig.savefig(save_folder + "ZFish_C{0}_vs_CElegans_C{1}.pdf".format(rw, cl), type="pdf")
