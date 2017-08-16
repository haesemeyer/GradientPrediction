#  Copyright 2017 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Module to generate, save and load training and test data and provide access to that data in convenient batches
"""

import numpy as np
import h5py
from core import FRAME_RATE, HIST_SECONDS, PRED_WINDOW


class RandCash:
    """
    Represents cash of random numbers to draw from
    """
    def __init__(self, init_size, f_rand, max_size=10000000):
        """
        Creates a new RandCash
        :param init_size: The initial cash size
        :param f_rand: The function to obtain random numbers should only take size parameter
        :param max_size: The maximal size of the cash, defaults to 10e6
        """
        self.max_size = max_size
        self.__cash = f_rand(init_size)
        self.__cnt = -1
        self.__f_rand = f_rand

    def next_rand(self):
        """
        Returns the next number from the cash
        """
        self.__cnt += 1
        if self.__cnt < self.__cash.size:
            return self.__cash[self.__cnt]
        else:
            self.__cnt = 0
            self.__cash = self.__f_rand(min(self.__cash.size * 2, self.max_size))
            return self.__cash[self.__cnt]


class GradientData:
    """
    Class that represents training/test data from a gradient experiment
    """
    pass


class GradientSimulation:
    """
    Class to run a gradient simulation, creating training data in the process
    The simulation will at each timepoint simulate what could have happened
    500 ms into the future for each selectable behavior but only one path
    will actually be chosen to advance the simulation to avoid massive branching
    """
    def __init__(self, nsteps: int, radius, t_min, t_max):
        """
        Creates a new GradientSimulation
        :param nsteps: The number of steps to perform
        :param radius: The arena radius in mm
        :param t_min: The center temperature
        :param t_max: The edge temperature
        """
        self.nsteps = nsteps
        self.radius = radius
        self.t_min = t_min
        self.t_max = t_max
        # Create 3D array to hold all model inputs derived during the simulation
        self.model_inputs = np.empty((nsteps, 3, FRAME_RATE*HIST_SECONDS), dtype=np.float32)
        # Create 2D array to hold all possible final temperatures during the simulation
        self.Temp_out = np.empty((nsteps, 4), dtype=np.float32)
        # Set bout parameters used in the simulation
        self.p_move = 1.0 / FRAME_RATE  # Bout frequency of 1Hz on average
        self.blen = int(FRAME_RATE * 0.2)  # Bouts happen over 200 ms length
        self.bfrac = np.linspace(0, 1, self.blen)
        # Displacement is drawn from gamma distribution
        self.disp_k = 2.63
        self.disp_theta = 1 / 0.138
        # Turn angles of straight swims and turns are drawn from gaussian
        self.mu_str = np.deg2rad(0)
        self.sd_str = np.deg2rad(2)
        self.mu_trn = np.deg2rad(30)
        self.sd_trn = np.deg2rad(5)
        # set up cashes of random numbers for bout parameters - divisor for disp is for conversion to mm
        self.__disp_cash = RandCash(1000, lambda s: np.random.gamma(self.disp_k, self.disp_theta, s) / 9)
        self.__str_cash = RandCash(1000, lambda s: np.random.randn(s) * self.sd_str + self.mu_str)
        self.__trn_cash = RandCash(1000, lambda s: np.random.randn(s) * self.sd_trn + self.mu_trn)
        self.__uni_cash = RandCash(1000, lambda s: np.random.rand(s))
        # place holder to receive bout trajectories for efficiency
        self.__bout = np.empty((self.blen, 3))

    def temperature(self, r):
        """
        Returns the temperature at the given radius
        """
        return (r / self.radius) * (self.t_max - self.t_min) + self.t_min

    def get_bout_type(self):
        """
        With 1/3 probability for each type returns a random bout type
        """
        dec = self.__uni_cash.next_rand()
        if dec < 1.0/3:
            return "S"
        elif dec < 2.0/3:
            return "L"
        else:
            return "R"

    def get_bout_trajectory(self, start, bout_type="S"):
        """
        Gets a trajectory for the given bout type
        :param start: Tuple/vector of x, y, angle at start of bout
        :param bout_type: The type of bout: (S)traight, (L)eft turn, (R)ight turn
        :return: The trajectory of the bout (blen rows, 3 columns: x, y, angle)
        """
        if bout_type == "S":
            da = self.__str_cash.next_rand()
        elif bout_type == "L":
            da = -1 * self.__trn_cash.next_rand()
        elif bout_type == "R":
            da = self.__trn_cash.next_rand()
        else:
            raise ValueError("bout_type has to be one of S, L, or R")
        heading = start[2] + da
        disp = self.__disp_cash.next_rand()
        dx = np.cos(heading) * disp * self.bfrac
        dy = np.sin(heading) * disp * self.bfrac
        # reflect bout if it would take us outside the dish
        r_end = np.sqrt((start[0]+dx)**2 + (start[1]+dy)**2)[-1]
        if r_end > self.radius:
            heading = heading + np.pi
            dx = np.cos(heading) * disp * self.bfrac
            dy = np.sin(heading) * disp * self.bfrac
        self.__bout[:, 0] = dx + start[0]
        self.__bout[:, 1] = dy + start[1]
        self.__bout[:, 2] = heading
        return self.__bout

    def run_simulation(self):
        """
        Forward run of random gradient exploration
        :return: The position and heading in the gradient at each timepoint
        """
        all_pos = np.zeros((self.nsteps, 3))
        i = 1
        while i < self.nsteps:
            dec = self.__uni_cash.next_rand()
            if dec < self.p_move:
                bt = self.get_bout_type()
                traj = self.get_bout_trajectory(all_pos[i-1, :], bt)
                if i+self.blen <= self.nsteps:
                    all_pos[i:i+self.blen, :] = traj
                else:
                    all_pos[i:, :] = traj[:all_pos[i:, :].shape[0], :]
                i += self.blen
            else:
                all_pos[i, :] = all_pos[i-1, :]
                i += 1
        return all_pos


if __name__ == '__main__':
    pass
