#  Copyright 2017 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script to simulate gradient navigation by a trained prediction model
"""

import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
import sys
import matplotlib as mpl
from core import ModelData, GradientData, ModelSimulation, ZfGpNetworkModel, CircleGradSimulation
from core import LinearGradientSimulation


def run_simulation(simulation: ModelSimulation, n_steps, run_ideal=False, simdir="r"):
    """
    Run a neural network model simulation computing occupancy histograms
    :param simulation: The simulation to run
    :param n_steps: The number of steps to simulation
    :param run_ideal: If True, instead of using neural network movement will be based on true ideal choices
    :param simdir: Determines whether occupancy should be calculated along (r)adius, (x)- or (y)-axis
    :return:
        [0]: All positions
        [1]: bin_centers in degree celcius
        [2]: Relative occupancy (corrected if radial)
    """
    if simdir not in ["r", "x", "y"]:
        raise ValueError("simdir has to be one of (r)adius, (x)- or (y)-axis")
    if run_ideal:
        pos = simulation.run_ideal(n_steps)
    else:
        pos = simulation.run_simulation(n_steps)
    if simdir == "r":
        quantpos = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
    elif simdir == "x":
        quantpos = pos[:, 0]
    else:
        quantpos = pos[:, 1]
    bins = np.linspace(0, simulation.max_pos, 100)
    bin_centers = bins[:-1] + np.diff(bins) / 2
    h = np.histogram(quantpos, bins)[0].astype(float)
    # for radial histogram normalize by radius to offset area increase
    if simdir == "r":
        h = h / bin_centers
    h = h / h.sum()
    if simdir == "r" or simdir == "x":
        bin_centers = simulation.temperature(bin_centers, np.zeros_like(bin_centers))
    else:
        bin_centers = simulation.temperature(np.zeros_like(bin_centers), bin_centers)
    return pos, bin_centers, h


if __name__ == "__main__":
    if sys.platform == "darwin" and "Tk" not in mpl.get_backend():
        print("On OSX tkinter likely does not work properly if matplotlib uses a backend that is not TkAgg!")
        print("If using ipython activate TkAgg backend with '%matplotlib tk' and retry.")
        sys.exit(1)
    n_steps = 2000000
    TPREFERRED = 25
    root = tk.Tk()
    root.update()
    root.withdraw()
    print("Select model directory")
    model_dir = filedialog.askdirectory(title="Select directory with model checkpoints", initialdir="./model_data/")
    root.update()
    mdata = ModelData(model_dir)
    # load training data for scaling
    try:
        std = GradientData.load_standards("gd_training_data.hdf5")
    except IOError:
        print("No standards found attempting to load full training data")
        std = GradientData.load("gd_training_data.hdf5").standards
    sim_type = ""
    while sim_type != "l" and sim_type != "r":
        sim_type = input("Please select either (l)inear or (r)adial simulation [l/r]:")
    gpn_naive = ZfGpNetworkModel()
    gpn_naive.load(mdata.ModelDefinition, mdata.FirstCheckpoint)
    gpn_trained = ZfGpNetworkModel()
    gpn_trained.load(mdata.ModelDefinition, mdata.LastCheckpoint)
    if sim_type == "l":
        sim_type = "x"  # so we call run_simulation correctly later
        sim_naive = LinearGradientSimulation(gpn_naive, std, 100, 100, 22, 37, TPREFERRED)
        sim_trained = LinearGradientSimulation(gpn_trained, std, 100, 100, 22, 37, TPREFERRED)
    else:
        sim_naive = CircleGradSimulation(gpn_naive, std, 100, 22, 37, TPREFERRED)
        sim_trained = CircleGradSimulation(gpn_trained, std, 100, 22, 37, TPREFERRED)
    b_naive, h_naive = run_simulation(sim_naive, n_steps, False, sim_type)[1:]
    b_trained, h_trained = run_simulation(sim_trained, n_steps, False, sim_type)[1:]
    b_ideal, h_ideal = run_simulation(sim_trained, n_steps, True, sim_type)[1:]

    fig, ax = pl.subplots()
    ax.plot(b_naive, h_naive, label="Naive")
    ax.plot(b_trained, h_trained, label="Trained")
    ax.plot(b_ideal, h_ideal, label="Perfect")
    if TPREFERRED is not None:
        max_frac = np.max(np.r_[h_naive, h_trained, h_ideal])
        ax.plot([TPREFERRED, TPREFERRED], [0, max_frac], 'k--')
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel("Occupancy")
    ax.set_ylim(0)
    ax.legend()
    sns.despine(fig, ax)
