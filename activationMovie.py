#  Copyright 2017 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script to create movie frames of network activations upon temperature stimulation and behavior generation
"""


import sys
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as pl
import tkinter as tk
from tkinter import filedialog
from core import GradientData, ModelData, ZfGpNetworkModel
from trainingData import CircGradientTrainer
from global_defs import GlobalDefs


if __name__ == "__main__":
    if sys.platform == "darwin" and "Tk" not in mpl.get_backend():
        print("On OSX tkinter likely does not work properly if matplotlib uses a backend that is not TkAgg!")
        print("If using ipython activate TkAgg backend with '%matplotlib tk' and retry.")
        sys.exit(1)
    # load training data to obtain temperature scaling
    try:
        std = GradientData.load_standards("gd_training_data.hdf5")
    except IOError:
        print("No standards found attempting to load full training data")
        train_data = GradientData.load("gd_training_data.hdf5")
        std = train_data.standards
    # load and interpolate temperature stimulus
    dfile = h5py.File("stimFile.hdf5", 'r')
    tsin = np.array(dfile['sine_L_H_temp'])
    x = np.arange(tsin.size)  # stored at 20 Hz !
    xinterp = np.linspace(0, tsin.size, tsin.size * GlobalDefs.frame_rate // 20)
    temp = np.interp(xinterp, x, tsin)
    dfile.close()
    print("Select model directory")
    root = tk.Tk()
    root.update()
    root.withdraw()
    model_dir = filedialog.askdirectory(title="Select directory with model checkpoints", initialdir="./model_data/")
    mdata = ModelData(model_dir)
    root.update()
    # create our model and load from last checkpoint
    gpn = ZfGpNetworkModel()
    gpn.load(mdata.ModelDefinition, mdata.LastCheckpoint)
    # prepend lead-in to stimulus
    lead_in = np.full(gpn.input_dims[2] - 1, np.mean(temp[:10]))
    temp = np.r_[lead_in, temp]
    # run a short simulation to create some sample trajectories for speed and angle inputs
    sim = CircGradientTrainer(100, 22, 37)
    sim.p_move = 0.1 / GlobalDefs.frame_rate  # use reduced movement rate to aide visualization
    pos = sim.run_simulation(temp.size + 1)
    spd = np.sqrt(np.sum(np.diff(pos[:, :2], axis=0)**2, 1))
    da = np.diff(pos[:, 2])
    activities = gpn.unit_stimulus_responses(temp, spd, da, std)
    # make actual movie at five hertz, simply by skipping and also only create first repeat
    for i in range(activities['o'][0].shape[0] // 60):
        fig, ax = ZfGpNetworkModel.plot_network(activities, i * 20)
        fig.savefig("./networkMovie/{0}.png".format(i), type="png", dpi=150)
        pl.close(fig)
