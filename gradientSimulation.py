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
from core import ModelData, GradientData, ModelSimulation


class CircleGradSimulation(ModelSimulation):
    """
    Implements a nn-Model based gradient navigation simulation
    """
    def __init__(self, model_file, chkpoint, tdata, radius, t_min, t_max, t_preferred=None):
        """
        Creates a new ModelGradSimulation
        :param model_file: The model definition file to use in the simulation (.meta)
        :param chkpoint: The model checkpoint containing the trained data (.ckpt)
        :param tdata: Object that cotains training normalizations of model inputs
        :param radius: The arena radius
        :param t_min: The center temperature
        :param t_max: The edge temperature
        :param t_preferred: The preferred temperature or None to prefer minimum
        """
        super().__init__(model_file, chkpoint, tdata, t_preferred)
        self.radius = radius
        self.t_min = t_min
        self.t_max = t_max
        # set range of starting positions to more sensible default
        self.maxstart = self.radius

    def temperature(self, x, y):
        """
        Returns the temperature at the given positions
        """
        r = np.sqrt(x**2 + y**2)  # this is a circular arena so compute radius
        return (r / self.radius) * (self.t_max - self.t_min) + self.t_min

    def out_of_bounds(self, x, y):
        """
        Detects whether the given x-y position is out of the arena
        :param x: The x position
        :param y: The y position
        :return: True if the given position is outside the arena, false otherwise
        """
        # circular arena, compute radial position of point and compare to arena radius
        r = np.sqrt(x**2 + y**2)
        return r > self.radius


if __name__ == "__main__":
    if sys.platform == "darwin" and "Tk" not in mpl.get_backend():
        print("On OSX tkinter likely does not work properly if matplotlib uses a backend that is not TkAgg!")
        print("If using ipython activate TkAgg backend with '%matplotlib tk' and retry.")
        sys.exit(1)
    TPREFERRED = 25
    root = tk.Tk()
    root.update()
    root.withdraw()
    print("Select model directory")
    model_dir = filedialog.askdirectory(title="Select directory with model checkpoints", initialdir="./model_data/")
    mdata = ModelData(model_dir)
    train_data = GradientData.load("gd_training_data.hdf5")
    model_sim = CircleGradSimulation(mdata.ModelDefinition, mdata.FirstCheckpoint, train_data, 100, 22, 37, TPREFERRED)
    pos_naive = model_sim.run_simulation(2000000)
    model_sim = CircleGradSimulation(mdata.ModelDefinition, mdata.LastCheckpoint, train_data, 100, 22, 37, TPREFERRED)
    pos_trained = model_sim.run_simulation(2000000)
    # run an "ideal" simulation for comparison
    pos_ideal = model_sim.run_ideal(2000000, 0.0)
    r_naive = np.sqrt(pos_naive[:, 0]**2 + pos_naive[:, 1]**2)
    r_trained = np.sqrt(pos_trained[:, 0]**2 + pos_trained[:, 1]**2)
    r_ideal = np.sqrt(pos_ideal[:, 0]**2 + pos_ideal[:, 1]**2)
    # generate histograms
    bins = np.linspace(0, 100, 100)
    bcenters = bins[:-1] + np.diff(bins) / 2
    tbc = model_sim.temperature(0, bcenters)
    h_naive = np.histogram(r_naive, bins)[0]
    h_naive = h_naive / tbc
    h_naive /= h_naive.sum()
    h_trained = np.histogram(r_trained, bins)[0]
    h_trained = h_trained / tbc
    h_trained /= h_trained.sum()
    h_ideal = np.histogram(r_ideal, bins)[0]
    h_ideal = h_ideal / tbc
    h_ideal /= h_ideal.sum()

    fig, ax = pl.subplots()
    ax.plot(tbc, h_naive, label="Naive")
    ax.plot(tbc, h_trained, label="Trained")
    ax.plot(tbc, h_ideal, label="Ideal choice")
    if TPREFERRED is not None:
        max_frac = np.max(np.r_[h_naive, h_trained, h_ideal])
        ax.plot([TPREFERRED, TPREFERRED], [0, max_frac], 'k--')
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel("Occupancy")
    ax.legend()
    sns.despine(fig, ax)
