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
    def __init__(self, nsteps: int):
        """
        Creates a new GradientSimulation
        :param nsteps: The number of steps to perform
        """
        self.nsteps = nsteps
        # Create 3D array to hold all model inputs derived during the simulation
        self.model_inputs = np.empty((nsteps, 3, FRAME_RATE*HIST_SECONDS), dtype=np.float32)
        # Create 2D array to hold all possible final temperatures during the simulation
        self.Temp_out = np.empty((nsteps, 4), dtype=np.float32)
        # Set bout parameters used in the simulation
        self.p_move = 1.0 / FRAME_RATE  # Bout frequency of 1Hz on average
        self.blen = int(FRAME_RATE * 0.2)  # Bouts happen over 200 ms length
        # Displacement is drawn from gamma distribution
        self.disp_k = 2.63
        self.disp_theta = 1 / 0.138
        # Turn angles of straight swims and turns are drawn from gaussian
        self.mu_str = 0
        self.sd_str = 2
        self.mu_trn = 30
        self.sd_trn = 5
        # set up cashes of random numbers for bout parameters
        self.__disp_cash = RandCash(1000, lambda s: np.random.gamma(self.disp_k, self.disp_theta, s))
        self.__str_cash = RandCash(1000, lambda s: np.random.randn(s) * self.sd_str + self.mu_str)
        self.__trn_cash = RandCash(1000, lambda s: np.random.randn(s) * self.sd_trn + self.mu_trn)
        self.__uni_cash = RandCash(1000, lambda s: np.random.rand(s))


if __name__ == '__main__':
    pass
