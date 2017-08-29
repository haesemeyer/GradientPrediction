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
    def __init__(self, model_file, chkpoint, tdata, radius, t_min, t_max):
        """
        Creates a new ModelGradSimulation
        :param model_file: The model definition file to use in the simulation (.meta)
        :param chkpoint: The model checkpoint containing the trained data (.ckpt)
        :param tdata: Object that cotains training normalizations of model inputs
        :param radius: The arena radius
        :param t_min: The center temperature
        :param t_max: The edge temperature
        """
        super().__init__(0, radius, t_min, t_max)
        self.model_file = model_file
        self.chkpoint = chkpoint
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
                r = np.sqrt(np.sum(pos[step - history:step, 0:2] ** 2, 1))
                model_in[0, 0, :, 0] = (self.temperature(r) - self.temp_mean) / self.temp_std
                spd = np.sqrt(np.sum(np.diff(pos[step - history - 1:step, 0:2], axis=0) ** 2, 1))
                model_in[0, 1, :, 0] = (spd - self.disp_mean) / self.disp_std
                dang = np.diff(pos[step - history - 1:step, 2], axis=0)
                model_in[0, 2, :, 0] = (dang - self.ang_mean) / self.ang_std
                # to favor behavior towards center put action that results in lowest temperature first
                behav_ranks = np.argsort(m_out.eval(feed_dict={x_in: model_in, keep_prob: 1.0}).ravel())
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
                t_out[i] = self.temperature(np.sqrt(fpos[0] ** 2 + fpos[1] ** 2))
            # to favor behavior towards center put action that results in lowest temperature first
            behav_ranks = np.argsort(t_out).ravel()
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
    root = tk.Tk()
    root.update()
    root.withdraw()
    print("Select model directory")
    model_dir = filedialog.askdirectory(title="Select directory with model checkpoints", initialdir="./model_data/")
    mdata = ModelData(model_dir)
    train_data = GradientData.load("gd_training_data.hdf5")
    model_sim = ModelGradSimulation(mdata.ModelDefinition, mdata.FirstCheckpoint, train_data, 100, 22, 37)
    pos_naive = model_sim.run_simulation(2000000)
    model_sim = ModelGradSimulation(mdata.ModelDefinition, mdata.LastCheckpoint, train_data, 100, 22, 37)
    pos_trained = model_sim.run_simulation(2000000)
    # run an "ideal" simulation but randomizing choice in 50% of cases
    pos_ideal = model_sim.run_ideal(2000000, 0.5)
    r_naive = np.sqrt(pos_naive[:, 0]**2 + pos_naive[:, 1]**2)
    r_trained = np.sqrt(pos_trained[:, 0]**2 + pos_trained[:, 1]**2)
    r_ideal = np.sqrt(pos_ideal[:, 0]**2 + pos_ideal[:, 1]**2)

    fig, ax = pl.subplots()
    sns.kdeplot(r_naive, ax=ax)
    sns.kdeplot(r_trained, ax=ax)
    sns.kdeplot(r_ideal, ax=ax)
    sns.despine(fig, ax)
