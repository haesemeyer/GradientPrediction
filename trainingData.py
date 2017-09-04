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
    def __init__(self, model_in, model_out, pred_window=PRED_WINDOW, frame_rate=FRAME_RATE, hist_seconds=HIST_SECONDS):
        """
        Creates a new GradientData object
        :param model_in: The input data for training
        :param model_out: The real output for training
        :param pred_window: The prediction window used in the simulation
        :param frame_rate: The frame rate used in the simulation
        :param hist_seconds: The length of history provided to the model in seconds
        """
        self.model_in = model_in
        self.model_out = model_out
        self.data_size = model_in.shape[0]
        self.rev_map = np.arange(self.data_size)  # nothing shuffled yet
        # randomly shuffle input data and store reverse mapping
        self.shuffle_data()
        # store other information
        self.pred_window = pred_window
        self.frame_rate = frame_rate
        self.hist_seconds = hist_seconds
        # compute normalization constants
        self.temp_mean = np.mean(self.model_in_raw[:, 0, :])
        self.temp_std = np.std(self.model_in_raw[:, 0, :])
        self.disp_mean = np.mean(self.model_in_raw[:, 1, :])
        self.disp_std = np.std(self.model_in_raw[:, 1, :])
        self.ang_mean = np.mean(self.model_in_raw[:, 2, :])
        self.ang_std = np.std(self.model_in_raw[:, 2, :])
        self.batch_start = 0

    def shuffle_data(self):
        """
        Shuffles the model data, storing reverse mapping for retrieval of data in original order
        """
        all_ix = np.arange(self.data_size)
        shuff_ix = np.random.choice(all_ix, self.data_size, False)
        self.model_in = self.model_in_raw[shuff_ix, :, :].copy()
        self.model_out = self.model_out_raw[shuff_ix, :].copy()
        self.rev_map = np.full(self.data_size, -1)
        for i in range(self.data_size):
            self.rev_map[shuff_ix[i]] = i

    def copy_normalization(self, gdata):
        """
        Copies normalization constants
        :param gdata: Another GradientData object from which to copy normalization constants
        """
        self.temp_mean = gdata.temp_mean
        self.temp_std = gdata.temp_std
        self.disp_mean = gdata.disp_mean
        self.disp_std = gdata.disp_std
        self.ang_mean = gdata.ang_mean
        self.ang_std = gdata.ang_std

    @property
    def model_in_raw(self):
        """
        The model in data in original order
        """
        return self.model_in[self.rev_map, :, :]

    @property
    def model_out_raw(self):
        """
        The model out data in original order
        """
        return self.model_out[self.rev_map, :]

    def zsc_inputs(self, m_in):
        """
        Return z-scored version of model input matrix
        :param m_in: The model input matrix
        :return: Column Zscored matrix
        """
        sub = np.r_[self.temp_mean, self.disp_mean, self.ang_mean][None, :, None]
        div = np.r_[self.temp_std, self.disp_std, self.ang_std][None, :, None]
        return (m_in - sub) / div

    def training_batch(self, batchsize):
        """
        Retrieves one training batch as a random sample from the underlying data
        :param batchsize:
        :return: tuple of inputs and outputs
        """
        batch_end = self.batch_start + batchsize
        if batch_end > self.data_size:
            # one epoch is done, reshuffle data and start over
            self.batch_start = 0
            self.shuffle_data()
            batch_end = batchsize
        m_in = self.model_in[self.batch_start:batch_end, :, :]
        m_o = (self.model_out[self.batch_start:batch_end, :] - self.temp_mean) / self.temp_std
        # update batch start for next call
        self.batch_start = batch_end
        return self.zsc_inputs(m_in)[:, :, :, None], m_o

    def save(self, filename, overwrite=False):
        """
        Saves the actual data to an hdf5 file
        :param filename: The file to save the data to
        :param overwrite: If true file will be overwritten if it exists
        """
        if overwrite:
            dfile = h5py.File(filename, 'w')
        else:
            dfile = h5py.File(filename, 'x')
        try:
            dfile.create_dataset("model_in_raw", data=self.model_in_raw)
            dfile.create_dataset("model_out_raw", data=self.model_out_raw)
            grp = dfile.create_group("model_info")
            grp.create_dataset("PRED_WINDOW", data=self.pred_window)
            grp.create_dataset("FRAME_RATE", data=self.frame_rate)
            grp.create_dataset("HIST_SECONDS", data=self.hist_seconds)
        finally:
            dfile.close()

    @staticmethod
    def load(filename):
        """
        Loads training data from an hdf5 file
        :param filename: The file to load data from
        :return: A GradientData object with the file data
        """
        dfile = h5py.File(filename, 'r')
        if "model_in_raw" not in dfile or "model_out_raw" not in dfile:
            dfile.close()
            raise IOError("File does not seem to contain gradient data")
        p = np.array(dfile["model_info"]["PRED_WINDOW"])
        f = np.array(dfile["model_info"]["FRAME_RATE"])
        h = np.array(dfile["model_info"]["HIST_SECONDS"])
        return GradientData(np.array(dfile["model_in_raw"]), np.array(dfile["model_out_raw"]), p, f, h)


class GradientSimulation:
    """
    Class to run a gradient simulation, creating training data in the process
    The simulation will at each timepoint simulate what could have happened
    500 ms into the future for each selectable behavior but only one path
    will actually be chosen to advance the simulation to avoid massive branching
    """
    def __init__(self, radius, t_min, t_max):
        """
        Creates a new GradientSimulation
        :param radius: The arena radius in mm
        :param t_min: The center temperature
        :param t_max: The edge temperature
        """
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
        self.avg_disp = self.disp_k * self.disp_theta
        # Turn angles of straight swims and turns are drawn from gaussian
        self.mu_str = np.deg2rad(0)
        self.sd_str = np.deg2rad(2)
        self.mu_trn = np.deg2rad(30)
        self.sd_trn = np.deg2rad(5)
        # set up cashes of random numbers for bout parameters - divisor for disp is for conversion to mm
        self._disp_cash = RandCash(1000, lambda s: np.random.gamma(self.disp_k, self.disp_theta, s) / 9)
        self._str_cash = RandCash(1000, lambda s: np.random.randn(s) * self.sd_str + self.mu_str)
        self._trn_cash = RandCash(1000, lambda s: np.random.randn(s) * self.sd_trn + self.mu_trn)
        self._uni_cash = RandCash(1000, lambda s: np.random.rand(s))
        # place holder to receive bout trajectories for efficiency
        self._bout = np.empty((self.blen, 3), np.float32)
        self._pos_cache = np.empty((1, 3), np.float32)

    def temperature(self, x, y):
        """
        Returns the temperature at the given positions
        """
        r = np.sqrt(x**2 + y**2)  # this is a circular arena so compute radius
        return (r / self.radius) * (self.t_max - self.t_min) + self.t_min

    def get_bout_type(self):
        """
        With 1/3 probability for each type returns a random bout type
        """
        dec = self._uni_cash.next_rand()
        if dec < 1.0/3:
            return "S"
        elif dec < 2.0/3:
            return "L"
        else:
            return "R"

    def get_bout_trajectory(self, start, bout_type="S", expected=False):
        """
        Gets a trajectory for the given bout type
        :param start: Tuple/vector of x, y, angle at start of bout
        :param bout_type: The type of bout: (S)traight, (L)eft turn, (R)ight turn
        :param expected: If true, instead of picking random bout pick the expected bout in the category
        :return: The trajectory of the bout (blen rows, 3 columns: x, y, angle)
        """
        if bout_type == "S":
            if expected:
                da = self.mu_str
            else:
                da = self._str_cash.next_rand()
        elif bout_type == "L":
            if expected:
                da = -1 * self.mu_trn
            else:
                da = -1 * self._trn_cash.next_rand()
        elif bout_type == "R":
            if expected:
                da = self.mu_trn
            else:
                da = self._trn_cash.next_rand()
        else:
            raise ValueError("bout_type has to be one of S, L, or R")
        heading = start[2] + da
        if expected:
            disp = self.avg_disp
        else:
            disp = self._disp_cash.next_rand()
        dx = np.cos(heading) * disp * self.bfrac
        dy = np.sin(heading) * disp * self.bfrac
        # reflect bout if it would take us outside the dish
        if self.out_of_bounds(start[0]+dx[-1], start[1]+dy[-1]):
            heading = heading + np.pi
            dx = np.cos(heading) * disp * self.bfrac
            dy = np.sin(heading) * disp * self.bfrac
        self._bout[:, 0] = dx + start[0]
        self._bout[:, 1] = dy + start[1]
        self._bout[:, 2] = heading
        return self._bout

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

    def run_simulation(self, nsteps):
        """
        Forward run of random gradient exploration
        :param nsteps: The number of steps to simulate
        :return: The position and heading in the gradient at each timepoint
        """
        return self.sim_forward(nsteps, np.zeros(3), "N").copy()

    def sim_forward(self, nsteps, start_pos, start_type):
        """
        Simulates a number of steps ahead
        :param nsteps: The number of steps to perform
        :param start_pos: The current starting conditions [x,y,a]
        :param start_type: The behavior to perform on the first step "N", "S", "L", "R"
        :return: The position at each timepoint nsteps*[x,y,a]
        """
        if start_type not in ["N", "S", "L", "R"]:
            raise ValueError("start_type has to be either (N)o bout, (S)traight, (L)eft or (R)right")
        if self._pos_cache.shape[0] != nsteps:
            self._pos_cache = np.zeros((nsteps, 3))
        all_pos = self._pos_cache
        if start_type == "N":
            all_pos[0, :] = start_pos
            i = 1
        else:
            # if a start bout should be drawn, draw the "expected" bout not a random one
            traj = self.get_bout_trajectory(start_pos, start_type, True)
            if traj.size <= nsteps:
                all_pos[:traj.shape[0], :] = traj
                i = traj.size
            else:
                return traj[:nsteps, :]
        while i < nsteps:
            dec = self._uni_cash.next_rand()
            if dec < self.p_move:
                bt = self.get_bout_type()
                traj = self.get_bout_trajectory(all_pos[i - 1, :], bt)
                if i + self.blen <= nsteps:
                    all_pos[i:i + self.blen, :] = traj
                else:
                    all_pos[i:, :] = traj[:all_pos[i:, :].shape[0], :]
                i += self.blen
            else:
                all_pos[i, :] = all_pos[i - 1, :]
                i += 1
        return all_pos

    def create_dataset(self, sim_pos):
        """
        Creates a GradientData object by executing all behavioral choices at simulated positions in which the fish
        was stationary
        :param sim_pos: Previously created simulation trajectory
        :return: GradientData object with all necessary training in- and outputs
        """
        if sim_pos.shape[1] != 3:
            raise ValueError("sim_pos has to be nx3 array with xpos, ypos and heading at each timepoint")
        history = FRAME_RATE * HIST_SECONDS
        start = history + 1  # start data creation with enough history present
        # initialize model inputs and outputs
        inputs = np.zeros((sim_pos.shape[0] - start, 3, history), np.float32)
        outputs = np.zeros((sim_pos.shape[0] - start, 4), np.float32)
        btypes = ["N", "S", "L", "R"]
        # create vector that tells us when the fish was moving
        all_dx = np.r_[0, np.diff(sim_pos[:, 0])]
        is_moving = all_dx != 0
        # loop over each position, simulating PRED_WINDOW into future to obtain real finish temperature
        for step in range(start, sim_pos.shape[0]):
            if is_moving[step]:
                continue
            # obtain inputs at given step
            inputs[step-start, 0, :] = self.temperature(sim_pos[step-history+1:step+1, 0],
                                                        sim_pos[step-history+1:step+1, 1])
            spd = np.sqrt(np.sum(np.diff(sim_pos[step-history:step+1, 0:2], axis=0)**2, 1))
            inputs[step-start, 1, :] = spd
            inputs[step-start, 2, :] = np.diff(sim_pos[step-history:step+1, 2], axis=0)
            # select each possible behavior in turn starting from this step and simulate
            # PRED_WINDOW steps into the future to obtain final temperature as output
            for i, b in enumerate(btypes):
                fpos = self.sim_forward(PRED_WINDOW, sim_pos[step, :], b)[-1, :]
                outputs[step-start, i] = self.temperature(fpos[0], fpos[1])
        # create gradient data object on all non-moving positions
        is_moving = is_moving[start:]
        assert is_moving.size == inputs.shape[0]
        return GradientData(inputs[np.logical_not(is_moving), :, :], outputs[np.logical_not(is_moving), :])


if __name__ == '__main__':
    import matplotlib.pyplot as pl
    import seaborn as sns
    response = ""
    while response not in ["y", "n"]:
        response = input("Run simulation with default arena? [y/n]:")
    if response == "y":
        nsteps = int(input("Number of steps to perform?"))
        gradsim = GradientSimulation(100, 22, 37)
        print("Running gradient simulation")
        pos = gradsim.run_simulation(nsteps)
        pl.figure()
        pl.plot(pos[:, 0], pos[:, 1])
        pl.xlabel("X position [mm]")
        pl.ylabel("Y position [mm]")
        sns.despine()
        print("Generating gradient data")
        grad_data = gradsim.create_dataset(pos)
        print("Done")
