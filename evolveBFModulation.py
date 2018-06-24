#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script to run an evolutionary algorithm to train bout frequency modulation in trained prediction networks
to improve targeting of 26C in a gradient
"""

import numpy as np
from core import ModelData, GradientData, ZfGpNetworkModel
from zf_simulators import BoutFrequencyEvolver
import os
from time import perf_counter


# file definitions
base_path = "./model_data/Adam_1e-4/sepInput_mixTrain/"
paths_512 = [f+'/' for f in os.listdir(base_path) if "_3m512_" in f]
# evolution globals
n_iter = 500000
n_gen = 50
TPREFERRED = 26

# Indicates the baseline bout frequency
TRAIN_BOUT_FREQ = 1


def mpath(path):
    return base_path + path[:-1]  # need to remove trailing slash


if __name__ == "__main__":
    # load training data for scaling - dependent on bout frequency
    if TRAIN_BOUT_FREQ == 1:
        std = GradientData.load_standards("gd_training_data.hdf5")
    elif TRAIN_BOUT_FREQ == 0.5:
        std = GradientData.load_standards("gd_05Hz_training_data.hdf5")
    elif TRAIN_BOUT_FREQ == 2:
        std = GradientData.load_standards("gd_2Hz_training_data.hdf5")
    else:
        raise Exception("No training data has been generated for the requested bout frequency")

    # evolve each 512 network unless it has been done before
    for p in paths_512:
        model_path = mpath(p)
        savedir = model_path + '/evolve/'
        if os.path.exists(savedir):
            print("Skipping evolution of {0} since output path already exists.".format(p), flush=True)
            continue
        print("Performing evolution of {0} for {1} generations.".format(p, n_gen), flush=True)
        os.makedirs(savedir)
        mdata = ModelData(model_path)
        gpn = ZfGpNetworkModel()
        gpn.load(mdata.ModelDefinition, mdata.LastCheckpoint)
        bfe = BoutFrequencyEvolver(std, gpn)
        # set baseline bout frequency according to constant above
        bfe.p_move *= TRAIN_BOUT_FREQ
        bfe.bf_mult = TRAIN_BOUT_FREQ
        weights = np.full((n_gen, bfe.n_networks, bfe.n_weights), np.nan)
        errors = np.full((n_gen, bfe.n_networks), np.nan)
        t_start = perf_counter()
        for gen in range(n_gen):
            # store weights and errors corresponding to these weights
            weights[gen, :, :] = bfe.weight_mat
            errors[gen, :] = bfe.evolve(n_iter)[0]
            np.save(savedir+"generation_weights.npy", weights)
            np.save(savedir+"generation_errors.npy", errors)
            print("Generation {0} of {1} on network {2} done. Average error {3} C.".format(gen, n_gen, p,
                                                                                           np.mean(errors[gen, :])),
                  flush=True)
            current_elapsed = perf_counter() - t_start
            time_per_iter = current_elapsed / (gen + 1)
            time_remaining = (n_gen - gen - 1) * time_per_iter
            h = time_remaining // 3600
            m = (time_remaining - h*3600) // 60
            print("{0}h and {1}m remaining".format(h, m))
