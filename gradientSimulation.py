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

    @property
    def max_pos(self):
        return self.radius


class LinearGradientSimulation(ModelSimulation):
    """
    Implements a nn-Model based linear gradient navigation simulation
    """
    def __init__(self, model_file, chkpoint, tdata, xmax, ymax, t_min, t_max, t_preferred=None):
        """
        Creates a new ModelGradSimulation
        :param model_file: The model definition file to use in the simulation (.meta)
        :param chkpoint: The model checkpoint containing the trained data (.ckpt)
        :param tdata: Object that cotains training normalizations of model inputs
        :param xmax: The maximum x-position (gradient direction)
        :param ymax: The maximum y-position (neutral direction)
        :param t_min: The x=0 temperature
        :param t_max: The x=xmax temperature
        :param t_preferred: The preferred temperature or None to prefer minimum
        """
        super().__init__(model_file, chkpoint, tdata, t_preferred)
        self.xmax = xmax
        self.ymax = ymax
        self.t_min = t_min
        self.t_max = t_max
        # set range of starting positions to more sensible default
        self.maxstart = max(self.xmax, self.ymax)

    def temperature(self, x, y):
        """
        Returns the temperature at the given positions
        """
        return (x / self.xmax) * (self.t_max - self.t_min) + self.t_min

    def out_of_bounds(self, x, y):
        """
        Detects whether the given x-y position is out of the arena
        :param x: The x position
        :param y: The y position
        :return: True if the given position is outside the arena, false otherwise
        """
        if x < 0 or x > self.xmax:
            return True
        if y < 0 or y > self.ymax:
            return True
        return False

    @property
    def max_pos(self):
        return self.xmax


def run_simulation(simulation: ModelSimulation, n_steps, run_ideal=False, simdir="r"):
    """
    Run a neural network model simulation computing occupancy histograms
    :param simulation: The simulation to run
    :param n_steps: The number of steps to simulation
    :param run_ideal: If True, instead of using neural network movement will be based on true ideal choices
    :param simdir: Determines whether occupancy should be calculated along (r)adius, (x)- or (y)-axis
    :return:
        [0]: All positions
        [1]: bin_centers
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
    train_data = GradientData.load("gd_training_data.hdf5")
    sim_type = ""
    while sim_type != "l" and sim_type != "r":
        sim_type = input("Please select either (l)inear or (r)adial simulation [l/r]:")
    if sim_type == "l":
        sim_type = "x"  # so we call run_simulation correctly later
        sim_naive = LinearGradientSimulation(mdata.ModelDefinition, mdata.FirstCheckpoint, train_data, 100, 100, 22, 37,
                                             TPREFERRED)
        sim_trained = LinearGradientSimulation(mdata.ModelDefinition, mdata.LastCheckpoint, train_data, 100, 100, 22,
                                               37, TPREFERRED)
    else:
        sim_naive = CircleGradSimulation(mdata.ModelDefinition, mdata.FirstCheckpoint, train_data, 100, 22, 37,
                                         TPREFERRED)
        sim_trained = CircleGradSimulation(mdata.ModelDefinition, mdata.LastCheckpoint, train_data, 100, 22, 37,
                                           TPREFERRED)
    b_naive, h_naive = run_simulation(sim_naive, n_steps, False, sim_type)[1:]
    b_trained, h_trained = run_simulation(sim_trained, n_steps, False, sim_type)[1:]
    b_ideal, h_ideal = run_simulation(sim_trained, n_steps, True, sim_type)[1:]

    fig, ax = pl.subplots()
    ax.plot(b_naive, h_naive, label="Naive")
    ax.plot(b_trained, h_trained, label="Trained")
    ax.plot(b_ideal, h_ideal, label="Ideal choice")
    if TPREFERRED is not None:
        max_frac = np.max(np.r_[h_naive, h_trained, h_ideal])
        ax.plot([TPREFERRED, TPREFERRED], [0, max_frac], 'k--')
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel("Occupancy")
    ax.legend()
    sns.despine(fig, ax)
