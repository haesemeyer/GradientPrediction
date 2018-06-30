#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script for panels of Figure S4 (Zebrafish phototaxis network)
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
from Figure3 import mpath
from scipy.signal import convolve
from sklearn.decomposition import PCA
from Figure4 import test_loss, plot_pc


# file definitions
base_path_zf = "./model_data/Adam_1e-4/sepInput_mixTrain/"
paths_512_zf = [f + '/' for f in os.listdir(base_path_zf) if "_3m512_" in f]

base_path_pt = "./model_data/Phototaxis/"
paths_512_pt = [f + '/' for f in os.listdir(base_path_pt) if "_3m512_" in f]


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
    save_folder = "./DataFigures/FigureS4/"
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

    std_zf = c.GradientData.load_standards("gd_training_data.hdf5")
    ana_zf = a.Analyzer(MoTypes(False), std_zf, "sim_store.hdf5", "activity_store.hdf5")
    std_pt = c.GradientData.load_standards("photo_training_data.hdf5")

    # load and interpolate temperature stimulus
    dfile = h5py.File("stimFile.hdf5", 'r')
    tsin = np.array(dfile['sine_L_H_temp'])
    x = np.arange(tsin.size)  # stored at 20 Hz !
    xinterp = np.linspace(0, tsin.size, tsin.size * GlobalDefs.frame_rate // 20)
    temperature = np.interp(xinterp, x, tsin)
    dfile.close()

    # get cell responses
    all_cells_zf = []
    for i, p in enumerate(paths_512_zf):
        cell_res, ids = ana_zf.temperature_activity(mpath(base_path_zf, p), temperature, i)
        all_cells_zf.append(cell_res)
    all_cells_zf = np.hstack(all_cells_zf)
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
