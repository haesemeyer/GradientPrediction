#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script to train gradient reinforcement learning navigation model
"""

from core import SimpleRLNetwork, TemperatureArena, FRAME_RATE, HIST_SECONDS
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
from scipy.ndimage import gaussian_filter1d


class RLTrainer(TemperatureArena):
    """
    Class for implementing training of a reinforcement learning model
    """
    def __init__(self, model: SimpleRLNetwork, t_preferred: float, t_mean: float, t_std: float):
        """
        Creates a new RLTrainer object
        :param model: The network model to be trained / to drive simulations
        :param t_preferred: The preferred temperature (for training purposes)
        :param t_mean: The average temperature in the arena
        :param t_std: The temperature standard deviation in the arena
        """
        super().__init__()
        self.model = model
        self.t_preferred = t_preferred
        self.t_mean = t_mean
        self.t_std = t_std
        self.btypes = ["S", "L", "R"]
        # all starting positions have to be within bounds but x and y coordinates are further limted to +/- maxstart
        self.maxstart = 10
        # possible unit removal
        self.remove = None

    def _standardize(self, temp_trace):
        """
        Takes a real temperature traces and zscores it
        :param temp_trace: The temperature trace
        :return: Z-scored version of the trace
        """
        return (temp_trace - self.t_mean)/self.t_std

    def get_bout_probability(self, x_in):
        """
        Returns the bout probability
        :param x_in: Model input - currently unused
        :return: The overall movement probability
        """
        return self.p_move

    def get_start_pos(self):
        """
        Returns a valid start position in the arena
        :return: 3-element vector (xpos, ypos, angle)
        """
        x = np.inf
        y = np.inf
        while self.out_of_bounds(x, y):
            x = np.random.randint(-self.maxstart, self.maxstart, 1)
            y = np.random.randint(-self.maxstart, self.maxstart, 1)
        a = np.random.rand() * 2 * np.pi
        return np.array([x, y, a])

    def select_behavior(self, selector):
        """
        Given a selector returns the actual bout type to perform (0="S", 1="L" or "R")
        """
        if selector == 0:
            return self.btypes[0]
        decider = self._uni_cash.next_rand()
        if decider < 0.5:
            return self.btypes[1]
        else:
            return self.btypes[2]

    @property
    def max_pos(self):
        return None

    def run_sim(self, nsteps, train=False):
        """
        Runs gradient simulation using the neural network model training with reinforcement after each behavior
        if train is set to true
        :param nsteps: The number of steps to simulate
        :param train: If true, the network will be trained after each step, otherwise just runs simulation
        :return:
            [0] Array of nsteps x 3 position arrays (xpos, ypos, angle)
            [1] For each performed  behavior the reward received
        """
        history = FRAME_RATE * HIST_SECONDS
        burn_period = history * 2
        start = history + 1
        pos = np.full((nsteps + burn_period, 3), np.nan)
        pos[:start + 1, :] = self.get_start_pos()[None, :]
        # list of collected rewards
        all_rewards = []
        # run simulation
        step = start
        model_in = np.zeros((1, 1, history, 1))
        # overall bout frequency at ~1 Hz
        last_p_move_evaluation = -100  # tracks the frame when we last updated our movement evaluation
        p_eval = self.p_move
        while step < nsteps + burn_period:
            # update our movement probability if necessary
            if step - last_p_move_evaluation >= 20:
                model_in[0, 0, :, 0] = self._standardize(self.temperature(pos[step - history:step, 0],
                                                                          pos[step - history:step, 1]))
                p_eval = self.get_bout_probability(model_in)
                last_p_move_evaluation = step
            if self._uni_cash.next_rand() > p_eval:
                pos[step, :] = pos[step - 1, :]
                step += 1
                continue
            model_in[0, 0, :, 0] = self._standardize(self.temperature(pos[step - history:step, 0],
                                                                      pos[step - history:step, 1]))
            chosen = self.model.choose_action(model_in, 0.01, 0.5 if train else 1.0, self.remove)
            bt = self.select_behavior(chosen)
            traj = self.get_bout_trajectory(pos[step - 1, :], bt)
            # compute reward
            pref = 0 if self.t_preferred is None else self.t_preferred
            d_start = np.abs(self.temperature(traj[0, 0], traj[0, 1]) - pref)
            d_end = np.abs(self.temperature(traj[-1, 0], traj[-1, 1]) - pref)
            reward = d_start - d_end
            all_rewards.append(reward)
            # train network
            if train:
                self.model.train(model_in, [reward], [chosen], 0.5)
            # implement trajectory
            if step + self.blen <= nsteps + burn_period:
                pos[step:step + self.blen, :] = traj
            else:
                pos[step:, :] = traj[:pos[step:, :].shape[0], :]
            step += self.blen
        return pos[burn_period:, :], all_rewards


class CircleRLTrainer(RLTrainer):
    """
    Implements a RL-Model based circular gradient navigation simulation and trainer
    """
    def __init__(self, model: SimpleRLNetwork, radius, t_min, t_max, t_preferred=None):
        """
        Creates a new ModelGradSimulation
        :param model: The network model to run the simulation
        :param radius: The arena radius
        :param t_min: The center temperature
        :param t_max: The edge temperature
        :param t_preferred: The preferred temperature or None to prefer minimum
        """
        # calculate expected temperature mean and standard deviation in this arena
        rads = np.linspace(0, radius, 500)
        temps = (rads / radius) * (t_max-t_min) + t_min
        t_mean = (temps*rads).sum() / rads.sum()
        t_std = np.sqrt((((temps-t_mean)**2)*rads).sum() / rads.sum())
        super().__init__(model, t_preferred, t_mean, t_std)
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


N_STEPS = 500000  # the number of time steps to run in each arena (NOTE: Not equal to number of generated behaviors)
N_EPOCHS = 25  # the number of total training epochs to run
N_CONV = 5  # the number of convolution filters in the network
N_LAYERS = 2  # the number of hidden layers in the network
N_UNITS = 64  # the number of units in each network hidden layer
T_PREFERRED = 26

if __name__ == "__main__":
    running_rewards = []  # for each performed step the received reward
    ep_avg_grad_error = []  # for each episode the average deviation from the preferred temperature
    with SimpleRLNetwork() as rl_net:
        rl_net.setup(N_CONV, N_UNITS, N_LAYERS)
        for ep in range(N_EPOCHS):
            circ_train = CircleRLTrainer(rl_net, 100, 22, 37, T_PREFERRED)
            ep_pos, ep_rewards = circ_train.run_sim(N_STEPS, True)
            temps = circ_train.temperature(ep_pos[:, 0], ep_pos[:, 1])
            avg_error = np.mean(np.abs(temps-T_PREFERRED))
            ep_avg_grad_error.append(avg_error)
            running_rewards += ep_rewards
            print("Epoch {0} of {1} has been completed. Average gradient error: {2}".format(ep+1, N_EPOCHS, avg_error))
        weights_conv1 = rl_net.convolution_data[0]
        weights_conv1 = weights_conv1['t']

    w_ext = np.max(np.abs(weights_conv1))
    fig, ax = pl.subplots(ncols=int(np.sqrt(N_CONV)), nrows=int(np.sqrt(N_CONV)), frameon=False,
                          figsize=(14, 2.8))
    ax = ax.ravel()
    for j, a in enumerate(ax):
        sns.heatmap(weights_conv1[:, :, 0, j], ax=a, vmin=-w_ext, vmax=w_ext, center=0, cbar=False)
        a.axis("off")

    ep_avg_grad_error = np.array(ep_avg_grad_error)
    running_rewards = np.array(running_rewards)
    fig, ax = pl.subplots()
    ax.plot(gaussian_filter1d(running_rewards, 250))
    ax.set_xlabel("Training step")
    ax.set_ylabel("Received rewards")
    sns.despine(fig, ax)

    fig, ax = pl.subplots()
    ax.plot(gaussian_filter1d(ep_avg_grad_error, 3), 'o')
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Average gradient error [C]")
    sns.despine(fig, ax)
