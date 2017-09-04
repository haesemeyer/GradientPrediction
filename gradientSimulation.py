#  Copyright 2017 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script to simulate gradient navigation by a trained prediction model
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
import sys
import matplotlib as mpl
from core import ModelData, FRAME_RATE, HIST_SECONDS
from trainingData import GradientSimulation, GradientData


class ModelGradSimulation(GradientSimulation):
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
        super().__init__(radius, t_min, t_max)
        self.model_file = model_file
        self.chkpoint = chkpoint
        self.t_preferred = t_preferred
        self.temp_mean = tdata.temp_mean
        self.temp_std = tdata.temp_std
        self.disp_mean = tdata.disp_mean
        self.disp_std = tdata.disp_std
        self.ang_mean = tdata.ang_mean
        self.ang_std = tdata.ang_std
        self.btypes = ["N", "S", "L", "R"]

    def get_start_pos(self):
        r = np.inf
        x = 0
        y = 0
        while r > self.radius**2:
            x = np.random.randint(-self.radius, self.radius, 1)
            y = np.random.randint(-self.radius, self.radius, 1)
            r = x**2 + y**2
        a = np.random.rand() * 2 * np.pi
        return np.array([x, y, a])

    def select_behavior(self, ranks):
        """
        Given a ranking of choices returns the bout type identifier to perform
        """
        decider = self._uni_cash.next_rand()
        if decider < 0.5:
            return self.btypes[ranks[0]]
        elif decider < 0.75:
            return self.btypes[ranks[1]]
        elif decider < 0.875:
            return self.btypes[ranks[2]]
        else:
            return self.btypes[ranks[3]]

    def run_simulation(self, nsteps):
        """
        Runs gradient simulation using the neural network model
        :param nsteps: The number of timesteps to perform
        :return: nsims long list of nsteps x 3 position arrays (xpos, ypos, angle)
        """
        history = FRAME_RATE * HIST_SECONDS
        burn_period = history * 2
        start = history + 1
        pos = np.full((nsteps + burn_period, 3), np.nan)
        pos[:start + 1, :] = self.get_start_pos()[None, :]
        # start session, load model and run simulation
        tf.reset_default_graph()
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(self.model_file)
            saver.restore(sess, self.chkpoint)
            graph = tf.get_default_graph()
            m_out = graph.get_tensor_by_name("m_out:0")
            x_in = graph.get_tensor_by_name("x_in:0")
            keep_prob = graph.get_tensor_by_name("keep_prob:0")
            step = start
            model_in = np.zeros((1, 3, history, 1))
            # overall bout frequency at ~1 Hz
            p_eval = 1.0 / FRAME_RATE
            while step < nsteps + burn_period:
                if self._uni_cash.next_rand() > p_eval:
                    pos[step, :] = pos[step-1, :]
                    step += 1
                    continue
                model_in[0, 0, :, 0] = (self.temperature(pos[step - history:step, 0], pos[step - history:step, 1])
                                        - self.temp_mean) / self.temp_std
                spd = np.sqrt(np.sum(np.diff(pos[step - history - 1:step, 0:2], axis=0) ** 2, 1))
                model_in[0, 1, :, 0] = (spd - self.disp_mean) / self.disp_std
                dang = np.diff(pos[step - history - 1:step, 2], axis=0)
                model_in[0, 2, :, 0] = (dang - self.ang_mean) / self.ang_std
                if self.t_preferred is None:
                    # to favor behavior towards center put action that results in lowest temperature first
                    behav_ranks = np.argsort(m_out.eval(feed_dict={x_in: model_in, keep_prob: 1.0}).ravel())
                else:
                    model_out = m_out.eval(feed_dict={x_in: model_in, keep_prob: 1.0}).ravel()
                    proj_diff = np.abs(model_out - (self.t_preferred - self.temp_mean)/self.temp_std)
                    behav_ranks = np.argsort(proj_diff)
                bt = self.select_behavior(behav_ranks)
                if bt == "N":
                    pos[step, :] = pos[step - 1, :]
                    step += 1
                    continue
                traj = self.get_bout_trajectory(pos[step-1, :], bt)
                if step + self.blen <= nsteps + burn_period:
                    pos[step:step + self.blen, :] = traj
                else:
                    pos[step:, :] = traj[:pos[step:, :].shape[0], :]
                step += self.blen

        return pos[burn_period:, :]

    def run_ideal(self, nsteps, pfail=0.0):
        """
        Runs gradient simulation picking the move that is truly ideal on average at each point
        :param nsteps: The number of timesteps to perform
        :param pfail: Probability of randomizing the order of behaviors instead of picking ideal
        :return: nsims long list of nsteps x 3 position arrays (xpos, ypos, angle)
        """
        from core import PRED_WINDOW
        history = FRAME_RATE * HIST_SECONDS
        burn_period = history * 2
        start = history + 1
        pos = np.full((nsteps + burn_period, 3), np.nan)
        pos[:start + 1, :] = self.get_start_pos()[None, :]
        step = start
        # overall bout frequency at ~1 Hz
        p_eval = 1.0 / FRAME_RATE
        t_out = np.zeros(4)
        while step < nsteps + burn_period:
            if self._uni_cash.next_rand() > p_eval:
                pos[step, :] = pos[step - 1, :]
                step += 1
                continue
            for i, b in enumerate(self.btypes):
                fpos = self.sim_forward(PRED_WINDOW, pos[step-1, :], b)[-1, :]
                t_out[i] = self.temperature(fpos[0], fpos[1])
            if self.t_preferred is None:
                # to favor behavior towards center put action that results in lowest temperature first
                behav_ranks = np.argsort(t_out).ravel()
            else:
                proj_diff = np.abs(t_out - self.t_preferred)
                behav_ranks = np.argsort(proj_diff).ravel()
            if self._uni_cash.next_rand() < pfail:
                np.random.shuffle(behav_ranks)
            bt = self.select_behavior(behav_ranks)
            if bt == "N":
                pos[step, :] = pos[step - 1, :]
                step += 1
                continue
            traj = self.get_bout_trajectory(pos[step - 1, :], bt)
            if step + self.blen <= nsteps + burn_period:
                pos[step:step + self.blen, :] = traj
            else:
                pos[step:, :] = traj[:pos[step:, :].shape[0], :]
            step += self.blen
        return pos[burn_period:, :]

    def create_dataset(self, sim_pos):
        raise NotImplementedError("This class does not implement this method")

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
    model_sim = ModelGradSimulation(mdata.ModelDefinition, mdata.FirstCheckpoint, train_data, 100, 22, 37, TPREFERRED)
    pos_naive = model_sim.run_simulation(2000000)
    model_sim = ModelGradSimulation(mdata.ModelDefinition, mdata.LastCheckpoint, train_data, 100, 22, 37, TPREFERRED)
    pos_trained = model_sim.run_simulation(2000000)
    # run an "ideal" simulation for comparison
    pos_ideal = model_sim.run_ideal(2000000, 0.0)
    r_naive = np.sqrt(pos_naive[:, 0]**2 + pos_naive[:, 1]**2)
    r_trained = np.sqrt(pos_trained[:, 0]**2 + pos_trained[:, 1]**2)
    r_ideal = np.sqrt(pos_ideal[:, 0]**2 + pos_ideal[:, 1]**2)
    # generate histograms
    bins = np.linspace(0, 100, 250)
    h_naive = np.histogram(r_naive, bins, weights=1.0/r_naive)[0]
    h_naive /= h_naive.sum()
    h_trained = np.histogram(r_trained, bins, weights=1.0/r_trained)[0]
    h_trained /= h_trained.sum()
    h_ideal = np.histogram(r_ideal, bins, weights=1.0/r_ideal)[0]
    h_ideal /= h_ideal.sum()

    fig, ax = pl.subplots()
    bcenters = bins[:-1] + np.diff(bins)/2
    tbc = model_sim.temperature(0, bcenters)
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
