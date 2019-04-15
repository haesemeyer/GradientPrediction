#  Copyright 2019 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Quantifies the "representational complexity" of trained and naive ANNs as the number of principal components
required to explain at least 99% of the total stimulus response variance in each network
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pl
import seaborn as sns
from mo_types import MoTypes
import core
import analysis
import h5py
from global_defs import GlobalDefs
from Figure3 import mpath
from sklearn.decomposition import PCA
from RL_trainingGround import CircleRLTrainer
from scipy.signal import convolve
from pandas import DataFrame
import os


# file definitions
base_path_zf = "./model_data/Adam_1e-4/sepInput_mixTrain/"
paths_512_zf = [f + '/' for f in os.listdir(base_path_zf) if "_3m512_" in f]

base_path_ce = "./model_data/CE_Adam_1e-4/"
paths_512_ce = [f + '/' for f in os.listdir(base_path_ce) if "_3m512_" in f]

base_path_pt = "./model_data/Phototaxis/"
paths_512_pt = [f + '/' for f in os.listdir(base_path_pt) if "_3m512_" in f]

base_path_th = "./model_data/Adam_1e-4/tanh/"
paths_512_th = [f + '/' for f in os.listdir(base_path_th) if "_3m512_" in f]

base_path_rl = "./model_data/FullRL_Net/"
paths_rl = [f + '/' for f in os.listdir(base_path_rl) if "mx_disc_" in f]


def get_cell_responses_predictive(path, stimulus, std: core.GradientStandards, trained=True):
    """
    Loads a model and computes the temperature response of all neurons returning response matrix
    :param path: Model path
    :param stimulus: Temperature stimulus
    :param std: Input standardizations
    :param trained: If false load naive network otherwise trained
    :return: n-timepoints x m-neurons matrix of responses
    """
    mdata = core.ModelData(path)
    # create our model and load from checkpoint
    gpn = core.ZfGpNetworkModel()
    gpn.load(mdata.ModelDefinition, mdata.LastCheckpoint if trained else mdata.FirstCheckpoint)
    # prepend lead-in to stimulus
    lead_in = np.full(gpn.input_dims[2] - 1, np.asscalar(np.mean(stimulus[:10])))
    temp = np.r_[lead_in, stimulus]
    activities = gpn.unit_stimulus_responses(temp, None, None, std)
    return np.hstack(activities['t']) if 't' in activities else np.hstack(activities['m'])


def get_cell_responses_rl(path, temp, temp_mean, temp_std, trained=True):
    """
    Loads a model and computes the temperature response of all neurons returning response matrix
    :param path: Model path
    :param temp: Temperature stimulus
    :param temp_mean: Average training temperature
    :param temp_std: Training temperature standard deviation
    :param trained: If false load naive network otherwise trained
    :return: n-timepoints x m-neurons matrix of responses
    """
    mdata = core.ModelData(path)
    rl = core.ReinforcementLearningNetwork()
    rl.load(mdata.ModelDefinition, mdata.LastCheckpoint if trained else mdata.FirstCheckpoint)
    # prepend lead-in to stimulus
    lead_in = np.full(rl.input_dims[2] - 1, np.asscalar(np.mean(temp[:10])))
    temp = np.r_[lead_in, temp]
    activities = rl.unit_stimulus_responses(temp, temp_mean, temp_std)
    return np.hstack(activities['t']) if 't' in activities else np.hstack(activities['m'])


def convolve_cell_responses(cell_mat, ca_kernel):
    """
    Convolves each response in cell_mat
    :param cell_mat: timeXunits response matrix
    :param ca_kernel: The calcium kernel for convolution
    :return: response matrix after convolution of each response with the calcium kernel
    """
    for cellnum in range(cell_mat.shape[1]):
        cell_mat[:, cellnum] = convolve(cell_mat[:, cellnum], ca_kernel, mode='full')[:cell_mat.shape[0]]


def complexity(cell_mat, v_c):
    """
    Computes the representational complexity across unit responses
    :param cell_mat: Responses of all network units
    :param v_c: The cumulative variance ratio cutoff
    :return: The number of principal components needed to explain at least v_c of variance
    """
    pca = PCA(20)
    pca.fit(cell_mat.T)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    return np.where(cum_var > v_c)[0][0] + 1  # index of first component is 0 in the array


if __name__ == "__main__":
    sns.reset_orig()
    mpl.rcParams['pdf.fonttype'] = 42

    var_cutoff = 0.99  # the fraction of variance to explain for complexity estimation

    std_zf = core.GradientData.load_standards("gd_training_data.hdf5")
    ana_zf = analysis.Analyzer(MoTypes(False), std_zf, "sim_store.hdf5", "activity_store.hdf5")
    std_ce = core.GradientData.load_standards("ce_gd_training_data.hdf5")
    ana_ce = analysis.Analyzer(MoTypes(True), std_ce, "ce_sim_store.hdf5", "ce_activity_store.hdf5")
    ana_th = analysis.Analyzer(MoTypes(False), std_zf, "sim_store_tanh.hdf5", "activity_store_tanh.hdf5")
    std_pt = core.GradientData.load_standards("photo_training_data.hdf5")

    # compute expected temperature mean and standard deviation for RL nets the same as during training
    ex1 = CircleRLTrainer(None, 100, 22, 37, 26)
    tm1 = ex1.t_mean
    s1 = ex1.t_std
    ex2 = CircleRLTrainer(None, 100, 14, 29, 26)
    tm2 = ex2.t_mean
    s2 = ex2.t_std
    rl_t_mean = (tm1 + tm2) / 2
    rl_t_std = (s1 + s2) / 2

    # load and interpolate temperature stimulus
    dfile = h5py.File("stimFile.hdf5", 'r')
    tsin = np.array(dfile['sine_L_H_temp'])
    x = np.arange(tsin.size)  # stored at 20 Hz !
    xinterp = np.linspace(0, tsin.size, tsin.size * GlobalDefs.frame_rate // 20)
    temperature = np.interp(xinterp, x, tsin)
    dfile.close()

    # generate calcium kernel
    tau_on = 1.4  # seconds
    tau_on *= GlobalDefs.frame_rate  # in frames
    tau_off = 2  # seconds
    tau_off *= GlobalDefs.frame_rate  # in frames
    kframes = np.arange(10 * GlobalDefs.frame_rate)  # 10 s long kernel
    kernel = 2 ** (-kframes / tau_off) * (1 - 2 ** (-kframes / tau_on))
    kernel = kernel / kernel.sum()

    complexity_dict = {"n_components": [], "net type": [], "trained": []}

    for i, p in enumerate(paths_512_zf):
        model_path = mpath(base_path_zf, p)
        cell_res = ana_zf.temperature_activity(model_path, temperature, i)[0]
        convolve_cell_responses(cell_res, kernel)
        comp_trained = complexity(cell_res, var_cutoff)
        cell_res = get_cell_responses_predictive(model_path, temperature, std_zf, False)
        convolve_cell_responses(cell_res, kernel)
        comp_naive = complexity(cell_res, var_cutoff)
        print("ZF Temp network: Trained: {0}, Naive: {1}".format(comp_trained, comp_naive))
        complexity_dict["n_components"].append(comp_trained)
        complexity_dict["trained"].append(True)
        complexity_dict["n_components"].append(comp_naive)
        complexity_dict["trained"].append(False)
        complexity_dict["net type"] += ["ZF Temp"]*2

    for i, p in enumerate(paths_512_ce):
        model_path = mpath(base_path_ce, p)
        cell_res = ana_ce.temperature_activity(model_path, temperature, i)[0]
        convolve_cell_responses(cell_res, kernel)
        comp_trained = complexity(cell_res, var_cutoff)
        cell_res = get_cell_responses_predictive(model_path, temperature, std_ce, False)
        convolve_cell_responses(cell_res, kernel)
        comp_naive = complexity(cell_res, var_cutoff)
        print("CE Temp network: Trained: {0}, Naive: {1}".format(comp_trained, comp_naive))
        complexity_dict["n_components"].append(comp_trained)
        complexity_dict["trained"].append(True)
        complexity_dict["n_components"].append(comp_naive)
        complexity_dict["trained"].append(False)
        complexity_dict["net type"] += ["CE Temp"]*2

    for i, p in enumerate(paths_512_th):
        model_path = mpath(base_path_th, p)
        cell_res = ana_th.temperature_activity(model_path, temperature, i)[0]
        convolve_cell_responses(cell_res, kernel)
        comp_trained = complexity(cell_res, var_cutoff)
        cell_res = get_cell_responses_predictive(model_path, temperature, std_zf, False)
        convolve_cell_responses(cell_res, kernel)
        comp_naive = complexity(cell_res, var_cutoff)
        print("Tanh network: Trained: {0}, Naive: {1}".format(comp_trained, comp_naive))
        complexity_dict["n_components"].append(comp_trained)
        complexity_dict["trained"].append(True)
        complexity_dict["n_components"].append(comp_naive)
        complexity_dict["trained"].append(False)
        complexity_dict["net type"] += ["Tanh"]*2

    for i, p in enumerate(paths_512_pt):
        model_path = mpath(base_path_pt, p)
        cell_res = get_cell_responses_predictive(model_path, temperature, std_pt, True)
        convolve_cell_responses(cell_res, kernel)
        comp_trained = complexity(cell_res, var_cutoff)
        cell_res = get_cell_responses_predictive(model_path, temperature, std_pt, False)
        convolve_cell_responses(cell_res, kernel)
        comp_naive = complexity(cell_res, var_cutoff)
        print("PT network: Trained: {0}, Naive: {1}".format(comp_trained, comp_naive))
        complexity_dict["n_components"].append(comp_trained)
        complexity_dict["trained"].append(True)
        complexity_dict["n_components"].append(comp_naive)
        complexity_dict["trained"].append(False)
        complexity_dict["net type"] += ["Phototaxis"]*2

    for i, p in enumerate(paths_rl):
        model_path = mpath(base_path_rl, p)
        cell_res = get_cell_responses_rl(model_path, temperature, rl_t_mean, rl_t_std, True)
        convolve_cell_responses(cell_res, kernel)
        comp_trained = complexity(cell_res, var_cutoff)
        cell_res = get_cell_responses_rl(model_path, temperature, rl_t_mean, rl_t_std, False)
        convolve_cell_responses(cell_res, kernel)
        comp_naive = complexity(cell_res, var_cutoff)
        print("RL network: Trained: {0}, Naive: {1}".format(comp_trained, comp_naive))
        complexity_dict["n_components"].append(comp_trained)
        complexity_dict["trained"].append(True)
        complexity_dict["n_components"].append(comp_naive)
        complexity_dict["trained"].append(False)
        complexity_dict["net type"] += ["RL"] * 2

    df_complexity = DataFrame(complexity_dict)

    fig, ax = pl.subplots()
    sns.barplot("net type", "n_components", "trained", df_complexity, ax=ax, ci=68, order=["ZF Temp", "Tanh", "CE Temp",
                                                                                           "RL", "Phototaxis"])
    sns.despine(fig, ax)
