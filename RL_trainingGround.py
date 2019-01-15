#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script to train gradient reinforcement learning navigation model
"""

from collections import deque
from core import SimpleRLNetwork
from zf_simulators import TemperatureArena
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import h5py
from global_defs import GlobalDefs
import os
from collections import Counter


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
        self.btypes = ["N", "S", "L", "R"]
        # all starting positions have to be within bounds but x and y coordinates are further limted to +/- maxstart
        self.maxstart = 10
        # possible unit removal
        self.remove = None
        # explore probabbility
        self.p_explore = 0.25
        # members for reward discounting
        self.discount_steps = 10
        self.discount_alpha = 0.01 ** (1/self.discount_steps)  # decay to 1% after discount_steps
        self.avg_reward = 0  # running average received reward
        self.rev_step = 1  # average reward calculation step counter

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
        Given a selector returns the actual bout type to perform (0="N", 1="S", 2="L" or "R")
        """
        return self.btypes[selector]

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
            [2] The selected behaviors (0-3)
        """
        train_memory = []
        history = GlobalDefs.frame_rate * GlobalDefs.hist_seconds
        burn_period = history * 2
        start = history + 1
        pos = np.full((nsteps + burn_period, 3), np.nan)
        pos[:start + 1, :] = self.get_start_pos()[None, :]
        # list of collected rewards
        all_rewards = []
        # list of selected behaviors
        all_behavs = []
        # run simulation
        step = start
        model_in = np.zeros((1, 1, history, 1))
        # overall bout frequency at ~1 Hz
        last_p_move_evaluation = -100  # tracks the frame when we last updated our movement evaluation
        p_eval = self.p_move
        # variables to prevent odd-ball behavior during training
        last_move_select = 0  # step in which network last moved (i.e. evaluation happened *and* selection was not "N")
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
            chosen, this_was_explore = self.model.choose_action(model_in, self.p_explore, keep=0.1 if train else 1.0,
                                                                det_drop=self.remove)
            bt = self.select_behavior(chosen)
            all_behavs.append(chosen)
            if bt == "N":
                pref = 0 if self.t_preferred is None else self.t_preferred
                d_start = np.abs(self.temperature(pos[step-1, 0], pos[step-1, 1]) - pref)
                # punish staying based on current position and time since last move since fish never fully rest
                this_reward = -d_start / 1000 - (step-last_move_select)*self.p_move/3
                traj = None
            else:
                last_move_select = step
                traj = self.get_bout_trajectory(pos[step - 1, :], bt)
                # compute reward
                pref = 0 if self.t_preferred is None else self.t_preferred
                d_start = np.abs(self.temperature(traj[0, 0], traj[0, 1]) - pref)
                d_end = np.abs(self.temperature(traj[-1, 0], traj[-1, 1]) - pref)
                reward_move = d_start - d_end
                this_reward = reward_move
            # add data to training batch memory
            if train:
                train_memory.append((model_in.copy(), this_reward, chosen, this_was_explore))
            # implement trajectory
            if bt == "N":
                pos[step, :] = pos[step-1, :]
                step += 1
            else:
                if step + self.blen <= nsteps + burn_period:
                    pos[step:step + self.blen, :] = traj
                else:
                    pos[step:, :] = traj[:pos[step:, :].shape[0], :]
                step += self.blen
        # train the model if requested - NOTE: This is currently inefficient as it doesn't aggregate into batches!
        if train:
            # calculate discounted values
            g = 0
            for sample_ix in reversed(range(len(train_memory))):
                mi, rew, ch, we = train_memory[sample_ix]
                g = self.discount_alpha*g + rew
                train_memory[sample_ix] = (mi, g, ch, we)
                self.avg_reward = self.avg_reward + (1 / self.rev_step) * (g - self.avg_reward)
                self.rev_step += 1
            ix = np.arange(len(train_memory))
            np.random.shuffle(ix)
            for sample_ix in ix:
                mi, val, ch, we = train_memory[sample_ix]
                if not we:  # don't train on exploratory moves
                    self.model.train(mi, np.array([val-self.avg_reward]), np.array([ch]), 0.5)
                    all_rewards.append(val-self.avg_reward)  # note that this is not aligned with actual behaviors!
        return pos[burn_period:, :], all_rewards, all_behavs


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


N_STEPS = 100000  # the number of time steps to run in each arena (NOTE: Not equal to number of generated behaviors)
STEP_INC = 500  # the total number of steps to run is N_STEPS + episode*STEP_INC
N_EPOCHS = 1500  # the number of total training epochs to run
N_CONV = 20  # the number of convolution filters in the network
N_LAYERS = 3  # the numer of hidden layers in the upper network branches
N_UNITS = 128  # the number of units in each network hidden layer
T_PREFERRED = 26  # the preferred temperature after training
SAVE_EVERY = 500  # save network state every this many episodes
NUM_NETS = 20  # the total number of networks to train

if __name__ == "__main__":
    # instantiate example trainer objects to allow for a fixed temperature mean and standard across
    # both temperature regimes
    ex1 = CircleRLTrainer(None, 100, 22, 37, T_PREFERRED)
    tm1 = ex1.t_mean
    s1 = ex1.t_std
    ex2 = CircleRLTrainer(None, 100, 14, 29, T_PREFERRED)
    tm2 = ex2.t_mean
    s2 = ex2.t_std
    temp_mean = (tm1 + tm2) / 2
    temp_std = (s1 + s2) / 2
    for net_num in range(NUM_NETS):
        train_folder = "./model_data/SimpleRL_Net/mx_disc_{0}".format(net_num)
        try:
            os.makedirs(train_folder, exist_ok=False)
        except OSError:
            print("Skipping training of {0} since output path already exists.".format(net_num), flush=True)
            continue
        chk_file = train_folder + "/simpleRLModel.ckpt"
        running_rewards = []  # for each performed step the received reward
        ep_avg_grad_error = []  # for each episode the average deviation from the preferred temperature
        rewards_given = []  # for each episode the number of rewards given in that episode
        global_count = 0
        avg_reward = 0
        rev_step = 1
        with SimpleRLNetwork() as rl_net:
            rl_net.setup(N_CONV, N_UNITS, N_LAYERS)
            # save naive model including full graph
            save_path = rl_net.save_state(chk_file, 0)
            print("Model saved in file: %s" % save_path)
            for ep in range(N_EPOCHS):
                if ep % 2 == 0:
                    circ_train = CircleRLTrainer(rl_net, 100, 22, 37, T_PREFERRED)
                    circ_train.t_mean = temp_mean
                    circ_train.t_std = temp_std
                    circ_train.avg_reward = avg_reward
                    circ_train.rev_step = rev_step
                else:
                    circ_train = CircleRLTrainer(rl_net, 100, 14, 29, T_PREFERRED)
                    circ_train.t_mean = temp_mean
                    circ_train.t_std = temp_std
                    circ_train.avg_reward = avg_reward
                    circ_train.rev_step = rev_step
                ep_pos, ep_rewards, ep_behavs = circ_train.run_sim(N_STEPS + ep*STEP_INC, True)
                # carry average reward over to next episode
                avg_reward = circ_train.avg_reward
                rev_step = circ_train.rev_step
                # compute navigation error
                temps = circ_train.temperature(ep_pos[:, 0], ep_pos[:, 1])
                weights = 1 / np.sqrt(np.sum(ep_pos[:, :2] ** 2, 1))
                weights[np.isinf(weights)] = 0  # occurs when 0,0 was picked as starting point only
                sum_of_weights = np.nansum(weights)
                weighted_sum = np.nansum(np.sqrt((temps - T_PREFERRED) ** 2) * weights)
                avg_error = weighted_sum / sum_of_weights
                ep_avg_grad_error.append(avg_error)
                rewards_given.append(len(ep_rewards))
                running_rewards += ep_rewards
                print("Epoch {0} of {1} has been completed. Type: {2} Average gradient error: {3}".format(ep+1,
                                                                                                          N_EPOCHS,
                                                                                                          ep % 2,
                                                                                                          np.round(
                                                                                                              avg_error,
                                                                                                              1)))
                cnt = Counter(ep_behavs)
                nb = sum(cnt.values())
                print("Selected behaviors %: Stay: {0}, Straight: {1}, Left: {2}, Right: {3}".format(
                    np.round(cnt[0]*100/nb),
                    np.round(cnt[1]*100/nb),
                    np.round(cnt[2]*100/nb),
                    np.round(cnt[3]*100/nb)))
                print("Average reward: {0}".format(avg_reward))
                print("")
                global_count += 1
                if global_count % SAVE_EVERY == 0:
                    save_path = rl_net.save_state(chk_file, global_count, False)
                    print("Model saved in file: %s" % save_path)
            save_path = rl_net.save_state(chk_file, global_count, False)
            print("Final model saved in file: %s" % save_path)
            weights_conv1 = rl_net.convolution_data[0]
            weights_conv1 = weights_conv1['t']
            # run one final test
            circ_train = CircleRLTrainer(rl_net, 100, 22, 37, T_PREFERRED)
            circ_train.p_explore = 0.5
            circ_train.t_mean = temp_mean
            circ_train.t_std = temp_std
            ep_pos, ep_rewards, ep_behavs = circ_train.run_sim(1000000, False)
            temps = circ_train.temperature(ep_pos[:, 0], ep_pos[:, 1])
            weights = 1 / np.sqrt(np.sum(ep_pos[:, :2] ** 2, 1))
            weights[np.isinf(weights)] = 0  # occurs when 0,0 was picked as starting point only
            sum_of_weights = np.nansum(weights)
            weighted_sum = np.nansum(np.sqrt((temps - T_PREFERRED) ** 2) * weights)
            avg_error = weighted_sum / sum_of_weights
            print("Final test navigation error = {0} C".format(np.round(avg_error, 1)))

        fig, ax = pl.subplots()
        ax.plot(ep_pos[:, 0], ep_pos[:, 1], lw=0.5, color=[0, 0, 0, 0.5])
        ax.plot(ep_pos[0, 0], ep_pos[0, 1], 'ro')
        sns.despine(fig, ax)

        ep_avg_grad_error = np.array(ep_avg_grad_error)
        running_rewards = np.array(running_rewards)
        rewards_given = np.array(rewards_given)

        fig, ax = pl.subplots()
        ax.plot(np.cumsum(rewards_given), ep_avg_grad_error, '.')
        ax.set_xlabel("# Received rewards")
        ax.set_ylabel("Average gradient error [C]")
        ax.set_title("Model {0}. Final gradient test error = {1} C.".format(net_num, np.round(avg_error, 1)))
        sns.despine(fig, ax)

        # save loss evaluations to file
        with h5py.File(train_folder + "/losses.hdf5", "x") as dfile:
            dfile.create_dataset("episodes", data=np.arange(N_EPOCHS))
            dfile.create_dataset("ep_avg_grad_error", data=ep_avg_grad_error)
            dfile.create_dataset("rewards_given", data=rewards_given)
            dfile.create_dataset("running_rewards", data=running_rewards)
