#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Classes for zebrafish behavior simulations
Behavioral distributions are derived from aggregated data across behavioral experiments in (Haesemeyer, 2015)
Note that behavior is merely approximated as the goal is not to derive a realistic simulation of zebrafish behavior
"""

import numpy as np
from core import RandCash, GradientData, GradientStandards, ZfGpNetworkModel, indexing_matrix
from global_defs import GlobalDefs


PRED_WINDOW = int(GlobalDefs.frame_rate * 0.5)  # the model should predict the temperature 500 ms into the future


class TemperatureArena:
    """
    Base class for behavioral simulations that happen in a
    virtual arena where temperature depends on position
    """
    def __init__(self):
        # Set bout parameters used in the simulation
        self.p_move = 1.0 / GlobalDefs.frame_rate  # Bout frequency of 1Hz on average
        self.blen = int(GlobalDefs.frame_rate * 0.2)  # Bouts happen over 200 ms length
        self.bfrac = np.linspace(0, 1, self.blen)
        # Displacement is drawn from gamma distribution
        self.disp_k = 2.63
        self.disp_theta = 1 / 0.138
        self.avg_disp = self.disp_k * self.disp_theta  # NOTE: This is not adjusted to mm - expected moves are exagg.
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

    def temperature(self, x, y, a=0):
        """
        Returns the temperature at the given positions
        """
        pass

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
        return False

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


class TrainingSimulation(TemperatureArena):
    """
    Base class for simulations that generate network training data
    """
    def __init__(self):
        super().__init__()

    def run_simulation(self, nsteps):
        """
        Forward run of random gradient exploration
        :param nsteps: The number of steps to simulate
        :return: The position and heading in the gradient at each timepoint
        """
        return self.sim_forward(nsteps, np.zeros(3), "N").copy()

    def create_dataset(self, sim_pos):
        """
        Creates a GradientData object by executing all behavioral choices at simulated positions in which the fish
        was stationary
        :param sim_pos: Previously created simulation trajectory
        :return: GradientData object with all necessary training in- and outputs
        """
        if sim_pos.shape[1] != 3:
            raise ValueError("sim_pos has to be nx3 array with xpos, ypos and heading at each timepoint")
        history = GlobalDefs.frame_rate * GlobalDefs.hist_seconds
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
                                                        sim_pos[step-history+1:step+1, 1],
                                                        sim_pos[step-history+1:step+1, 2])
            spd = np.sqrt(np.sum(np.diff(sim_pos[step-history:step+1, 0:2], axis=0)**2, 1))
            inputs[step-start, 1, :] = spd
            inputs[step-start, 2, :] = np.diff(sim_pos[step-history:step+1, 2], axis=0)
            # select each possible behavior in turn starting from this step and simulate
            # PRED_WINDOW steps into the future to obtain final temperature as output
            for i, b in enumerate(btypes):
                fpos = self.sim_forward(PRED_WINDOW, sim_pos[step, :], b)[-1, :]
                outputs[step-start, i] = self.temperature(fpos[0], fpos[1], fpos[2])
        # create gradient data object on all non-moving positions
        is_moving = is_moving[start:]
        assert is_moving.size == inputs.shape[0]
        return GradientData(inputs[np.logical_not(is_moving), :, :], outputs[np.logical_not(is_moving), :], PRED_WINDOW)


class ModelSimulation(TemperatureArena):
    """
    Base class for simulations that use trained networks
    to perform gradient navigation
    """
    def __init__(self, model: ZfGpNetworkModel, tdata, t_preferred):
        """
        Creates a new ModelSimulation
        :param model: The network model to run the simulation
        :param tdata: Training data or related object to supply scaling information
        :param t_preferred: The preferred temperature that should be reached during the simulation
        """
        super().__init__()
        self.model = model
        self.t_preferred = t_preferred
        self.temp_mean = tdata.temp_mean
        self.temp_std = tdata.temp_std
        self.disp_mean = tdata.disp_mean
        self.disp_std = tdata.disp_std
        self.ang_mean = tdata.ang_mean
        self.ang_std = tdata.ang_std
        self.btypes = ["N", "S", "L", "R"]
        # all starting positions have to be within bounds but x and y coordinates are further limted to +/- maxstart
        self.maxstart = 10
        # optionally holds a list of vectors to suppress activation in units that should be "ablated"
        self.remove = None
        # optionally weights that will transform the output of the temperature branch of the model into a bout frequency
        self.bf_weights = None

    def get_bout_probability(self, model_in):
        if self.bf_weights is None:
            return self.p_move
        temp_out = self.model.branch_output('t', model_in, self.remove).ravel()
        activation = np.sum(temp_out * self.bf_weights)
        # apply nonlinearity
        bfreq = 1 / (1 + np.exp(-activation))  # [0, 1]
        bfreq = (2 - 0.5) * bfreq + 0.5  # [0.5, 2]
        # return corresponding probability
        return bfreq / GlobalDefs.frame_rate

    def get_start_pos(self):
        x = np.inf
        y = np.inf
        while self.out_of_bounds(x, y):
            x = np.random.randint(-self.maxstart, self.maxstart, 1)
            y = np.random.randint(-self.maxstart, self.maxstart, 1)
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

    @property
    def max_pos(self):
        return None

    def run_simulation(self, nsteps, debug=False):
        """
        Runs gradient simulation using the neural network model
        :param nsteps: The number of timesteps to perform
        :param debug: If set to true function will return debug output
        :return:
            [0] nsims long list of nsteps x 3 position arrays (xpos, ypos, angle)
            [1] Returned if debug=True. Dictionary with vector of temps and matrix of predictions at each position
        """
        debug_dict = {}
        t_out = np.zeros(4)  # for debug purposes to run true outcome simulations forward
        if debug:
            # debug dict only contains information for timesteps which were select for possible movement!
            debug_dict["curr_temp"] = np.full(nsteps, np.nan)  # the current temperature at this position
            debug_dict["pred_temp"] = np.full((nsteps, 4), np.nan)  # the network predicted temperature for each move
            debug_dict["sel_behav"] = np.zeros(nsteps, dtype="U1")  # the actually selected move
            debug_dict["true_temp"] = np.full((nsteps, 4), np.nan)  # the temperature if each move is simulated
        history = GlobalDefs.frame_rate * GlobalDefs.hist_seconds
        burn_period = history * 2
        start = history + 1
        pos = np.full((nsteps + burn_period, 3), np.nan)
        pos[:start + 1, :] = self.get_start_pos()[None, :]
        # run simulation
        step = start
        model_in = np.zeros((1, 3, history, 1))
        # overall bout frequency at ~1 Hz
        last_p_move_evaluation = -100  # tracks the frame when we last updated our movement evaluation
        p_eval = self.p_move
        while step < nsteps + burn_period:
            # update our movement probability if necessary
            if step - last_p_move_evaluation >= 20:
                model_in[0, 0, :, 0] = (self.temperature(pos[step - history:step, 0], pos[step - history:step, 1])
                                        - self.temp_mean) / self.temp_std
                p_eval = self.get_bout_probability(model_in)
                last_p_move_evaluation = step
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
            model_out = self.model.predict(model_in, 1.0, self.remove).ravel()
            if self.t_preferred is None:
                # to favor behavior towards center put action that results in lowest temperature first
                behav_ranks = np.argsort(model_out)
            else:
                proj_diff = np.abs(model_out - (self.t_preferred - self.temp_mean)/self.temp_std)
                behav_ranks = np.argsort(proj_diff)
            bt = self.select_behavior(behav_ranks)
            if debug:
                dbpos = step - burn_period
                debug_dict["curr_temp"][dbpos] = model_in[0, 0, -1, 0] * self.temp_std + self.temp_mean
                debug_dict["pred_temp"][dbpos, :] = model_out * self.temp_std + self.temp_mean
                debug_dict["sel_behav"][dbpos] = bt
                for i, b in enumerate(self.btypes):
                    fpos = self.sim_forward(PRED_WINDOW, pos[step-1, :], b)[-1, :]
                    t_out[i] = self.temperature(fpos[0], fpos[1])
                debug_dict["true_temp"][dbpos, :] = t_out
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
        if debug:
            return pos[burn_period:, :], debug_dict
        return pos[burn_period:, :]

    def run_ideal(self, nsteps, pfail=0.0):
        """
        Runs gradient simulation picking the move that is truly ideal on average at each point
        :param nsteps: The number of timesteps to perform
        :param pfail: Probability of randomizing the order of behaviors instead of picking ideal
        :return: nsims long list of nsteps x 3 position arrays (xpos, ypos, angle)
        """
        history = GlobalDefs.frame_rate * GlobalDefs.hist_seconds
        burn_period = history * 2
        start = history + 1
        pos = np.full((nsteps + burn_period, 3), np.nan)
        pos[:start + 1, :] = self.get_start_pos()[None, :]
        step = start
        model_in = np.zeros((1, 3, history, 1))
        last_p_move_evaluation = -100  # tracks the frame when we last updated our movement evaluation
        # overall bout frequency at ~1 Hz
        p_eval = self.p_move
        t_out = np.zeros(4)
        while step < nsteps + burn_period:
            # update our movement probability if necessary
            if step - last_p_move_evaluation >= 20:
                model_in[0, 0, :, 0] = (self.temperature(pos[step - history:step, 0], pos[step - history:step, 1])
                                        - self.temp_mean) / self.temp_std
                p_eval = self.get_bout_probability(model_in)
                last_p_move_evaluation = step
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


class CircleGradSimulation(ModelSimulation):
    """
    Implements a nn-Model based gradient navigation simulation
    """
    def __init__(self, model: ZfGpNetworkModel, tdata, radius, t_min, t_max, t_preferred=None):
        """
        Creates a new ModelGradSimulation
        :param model: The network model to run the simulation
        :param tdata: Object that cotains training normalizations of model inputs
        :param radius: The arena radius
        :param t_min: The center temperature
        :param t_max: The edge temperature
        :param t_preferred: The preferred temperature or None to prefer minimum
        """
        super().__init__(model, tdata, t_preferred)
        self.radius = radius
        self.t_min = t_min
        self.t_max = t_max
        # set range of starting positions to more sensible default
        self.maxstart = self.radius

    def temperature(self, x, y, a=0):
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
    def __init__(self, model: ZfGpNetworkModel, tdata, xmax, ymax, t_min, t_max, t_preferred=None):
        """
        Creates a new ModelGradSimulation
        :param model: The network model to run the simulation
        :param tdata: Object that cotains training normalizations of model inputs
        :param xmax: The maximum x-position (gradient direction)
        :param ymax: The maximum y-position (neutral direction)
        :param t_min: The x=0 temperature
        :param t_max: The x=xmax temperature
        :param t_preferred: The preferred temperature or None to prefer minimum
        """
        super().__init__(model, tdata, t_preferred)
        self.xmax = xmax
        self.ymax = ymax
        self.t_min = t_min
        self.t_max = t_max
        # set range of starting positions to more sensible default
        self.maxstart = max(self.xmax, self.ymax)

    def temperature(self, x, y, a=0):
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


class PhototaxisSimulation(TemperatureArena):
    """
    Class to perform model guided simulation that attempts to keep
    facing a light source in a circular arena
    """
    def __init__(self, model: ZfGpNetworkModel, tdata, radius):
        """
        Creates a new ModelSimulation
        :param model: The network model to run the simulation
        :param tdata: Training data or related object to supply scaling information
        :param radius: The arena radius
        """
        super().__init__()
        self.model = model
        self.temp_mean = tdata.temp_mean
        self.temp_std = tdata.temp_std
        self.disp_mean = tdata.disp_mean
        self.disp_std = tdata.disp_std
        self.ang_mean = tdata.ang_mean
        self.ang_std = tdata.ang_std
        self.radius = radius
        self.maxstart = self.radius
        self.btypes = ["N", "S", "L", "R"]
        # all starting positions have to be within bounds but x and y coordinates are further limted to +/- maxstart
        self.maxstart = 10
        # optionally holds a list of vectors to suppress activation in units that should be "ablated"
        self.remove = None

    def get_start_pos(self):
        x = np.inf
        y = np.inf
        while self.out_of_bounds(x, y):
            x = np.random.randint(-self.maxstart, self.maxstart, 1)
            y = np.random.randint(-self.maxstart, self.maxstart, 1)
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

    @staticmethod
    def facing_angle(x, y, a=0):
        """
        Returns the angle to the central light-source at the given positions
        in a coordinate system centered on the object, facing in the heading direction
        """
        # calculate vector *to* source which is at coordinate 0/0
        vec_x = -x
        vec_y = -y
        # calculate heading unit vector
        head_x = np.cos(a)
        head_y = np.sin(a)
        # calculate and return angular position of source
        return np.arctan2(vec_y, vec_x) - np.arctan2(head_y, head_x)

    def run_simulation(self, nsteps, debug=False):
        """
        Runs gradient simulation using the neural network model
        :param nsteps: The number of timesteps to perform
        :param debug: If set to true function will return debug output
        :return:
            [0] nsims long list of nsteps x 3 position arrays (xpos, ypos, angle)
            [1] Returned if debug=True. Dictionary with vector of temps and matrix of predictions at each position
        """
        debug_dict = {}
        t_out = np.zeros(4)  # for debug purposes to run true outcome simulations forward
        if debug:
            # debug dict only contains information for timesteps which were select for possible movement!
            debug_dict["curr_angle"] = np.full(nsteps, np.nan)  # the current facing angle at this position
            debug_dict["pred_angle"] = np.full((nsteps, 4), np.nan)  # the network predicted angle for each move
            debug_dict["sel_behav"] = np.zeros(nsteps, dtype="U1")  # the actually selected move
            debug_dict["true_angle"] = np.full((nsteps, 4), np.nan)  # the facing angle if each move is simulated
        history = GlobalDefs.frame_rate * GlobalDefs.hist_seconds
        burn_period = history * 2
        start = history + 1
        pos = np.full((nsteps + burn_period, 3), np.nan)
        pos[:start + 1, :] = self.get_start_pos()[None, :]
        # run simulation
        step = start
        model_in = np.zeros((1, 3, history, 1))
        # overall bout frequency at ~1 Hz
        p_eval = self.p_move
        while step < nsteps + burn_period:
            if self._uni_cash.next_rand() > p_eval:
                pos[step, :] = pos[step-1, :]
                step += 1
                continue
            model_in[0, 0, :, 0] = (self.facing_angle(pos[step - history:step, 0], pos[step - history:step, 1],
                                                      pos[step - history:step, 2]) - self.temp_mean) / self.temp_std
            spd = np.sqrt(np.sum(np.diff(pos[step - history - 1:step, 0:2], axis=0) ** 2, 1))
            model_in[0, 1, :, 0] = (spd - self.disp_mean) / self.disp_std
            dang = np.diff(pos[step - history - 1:step, 2], axis=0)
            model_in[0, 2, :, 0] = (dang - self.ang_mean) / self.ang_std
            model_out = self.model.predict(model_in, 1.0, self.remove).ravel()
            # we rank the movements such that the highest rank [position 0] is given to the behavior that would
            # result in the smallest deviation from facing the light source - however this isn't neccesarily the
            # smallest angle - 10 degrees should be ranked worse then 359
            pred_angle = model_out * self.temp_std + self.temp_mean
            pred_angle %= (2*np.pi)  # ensure that no angles over 2pi remain
            # recode angles to smallest possible turn towards 0
            pred_angle[pred_angle > np.pi] = 2*np.pi - pred_angle[pred_angle > np.pi]
            pred_angle[pred_angle < -np.pi] = -2*np.pi - pred_angle[pred_angle < -np.pi]
            behav_ranks = np.argsort(np.abs(pred_angle))
            bt = self.select_behavior(behav_ranks)
            if debug:
                dbpos = step - burn_period
                debug_dict["curr_angle"][dbpos] = model_in[0, 0, -1, 0] * self.temp_std + self.temp_mean
                debug_dict["pred_angle"][dbpos, :] = model_out * self.temp_std + self.temp_mean
                debug_dict["sel_behav"][dbpos] = bt
                for i, b in enumerate(self.btypes):
                    fpos = self.sim_forward(PRED_WINDOW, pos[step-1, :], b)[-1, :]
                    t_out[i] = self.facing_angle(fpos[0], fpos[1], fpos[2])
                debug_dict["true_angle"][dbpos, :] = t_out
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
        if debug:
            return pos[burn_period:, :], debug_dict
        return pos[burn_period:, :]


class WhiteNoiseSimulation(TemperatureArena):
    """
    Class to perform white noise analysis of network models
    """
    def __init__(self, stds: GradientStandards, model: ZfGpNetworkModel, base_freq=1.0, stim_mean=0.0, stim_std=1.0):
        """
        Creates a new WhiteNoiseSimulation object
        :param stds: Standardization for displacement used in model training
        :param model: The network to use for the simulation (has to be initialized)
        :param base_freq: The baseline movement frequency in Hz
        :param stim_mean: The white noise stimulus average
        :param stim_std: The white noise stimulus standard deviation
        """
        super().__init__()
        self.p_move = base_freq / GlobalDefs.frame_rate
        self.model = model
        # for stimulus we do not generate real temperatures anyway so user selects parameters
        self.stim_mean = stim_mean
        self.stim_std = stim_std
        # for behaviors they need to be standardized according to training data
        self.disp_mean = stds.disp_mean
        self.disp_std = stds.disp_std
        self.ang_mean = stds.ang_mean
        self.ang_std = stds.ang_std
        self.btypes = ["N", "S", "L", "R"]
        # stimulus switching variables - if switch_mean<=0 then stimulus will be truly white, alternating randomly at
        # every frame. Otherwise switching times will be drawn from gaussian with mean switch_mean and sigma switch_std
        self.switch_mean = 0
        self.switch_std = 0
        # optional removal of network units
        self.remove = None
        # optionally weights that will transform the output of the temperature branch of the model into a bout frequency
        self.bf_weights = None

    # Private API
    def _get_bout_probability(self, model_in):
        if self.bf_weights is None:
            return self.p_move
        temp_out = self.model.branch_output('t', model_in, self.remove).ravel()
        activation = np.sum(temp_out * self.bf_weights)
        # apply nonlinearity
        bfreq = 1 / (1 + np.exp(-activation))  # [0, 1]
        bfreq = (2 - 0.5) * bfreq + 0.5  # [0.5, 2]
        # return corresponding probability
        return bfreq / GlobalDefs.frame_rate

    def _get_bout(self, bout_type: str):
        """
        For a given bout-type computes and returns the displacement and delta-heading trace
        """
        if bout_type not in self.btypes:
            raise ValueError("bout_type has to be one of {0}".format(self.btypes))
        if bout_type == "S":
            da = self._str_cash.next_rand()
        elif bout_type == "L":
            da = -1 * self._trn_cash.next_rand()
        else:
            da = self._trn_cash.next_rand()
        delta_heading = np.zeros(self.blen)
        delta_heading[0] = da
        disp = self._disp_cash.next_rand()
        displacement = np.full(self.blen, disp / self.blen)
        return displacement, delta_heading

    def _select_behavior(self, ranks):
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

    # Public API
    def compute_openloop_behavior(self, stimulus: np.ndarray):
        """
        Presents stimulus in open-loop to network and returns behavior traces
        :param stimulus: The stimulus to present to the network
        :return:
            [0]: Behavior instantiations (-1: none, 0: stay, 1: straight, 2: left, 3: right)
            [1]: Speed trace
            [2]: Angle trace
        """
        history = GlobalDefs.hist_seconds*GlobalDefs.frame_rate
        step = history
        model_in = np.zeros((1, 3, history, 1))
        behav_types = np.full(stimulus.size, -1, np.int8)
        speed_trace = np.zeros_like(stimulus)
        angle_trace = np.zeros_like(stimulus)
        last_p_move_evaluation = -100  # tracks the frame when we last updated our movement evaluation
        p_eval = self.p_move
        while step < stimulus.size:
            # update our movement probability if necessary
            if step - last_p_move_evaluation >= 20:
                model_in[0, 0, :, 0] = stimulus[step - history:step]
                p_eval = self._get_bout_probability(model_in)
                last_p_move_evaluation = step
            # first invoke the bout clock and pass if we shouldn't select a behavior
            if self._uni_cash.next_rand() > p_eval:
                step += 1
                continue
            model_in[0, 0, :, 0] = stimulus[step - history:step]
            model_in[0, 1, :, 0] = (speed_trace[step - history:step] - self.disp_mean) / self.disp_std
            model_in[0, 2, :, 0] = (angle_trace[step - history:step] - self.ang_mean) / self.ang_std
            model_out = self.model.predict(model_in, 1.0, self.remove).ravel()
            behav_ranks = np.argsort(model_out)
            bt = self._select_behavior(behav_ranks)
            if bt == "N":
                behav_types[step] = 0
                step += 1
                continue
            elif bt == "S":
                behav_types[step] = 1
            elif bt == "L":
                behav_types[step] = 2
            else:
                behav_types[step] = 3
            disp, dh = self._get_bout(bt)
            if step + self.blen <= stimulus.size:
                speed_trace[step:step + self.blen] = disp
                angle_trace[step:step + self.blen] = dh
            else:
                speed_trace[step:] = disp[:speed_trace[step:].size]
                angle_trace[step:] = dh[:angle_trace[step:].size]
            step += self.blen
        return behav_types, speed_trace, angle_trace

    def compute_behavior_kernels(self, n_samples=1e6):
        """
        Generates white-noise samples and presents them as an open-loop stimulus to the network computing filter kernels
        :param n_samples: The number of white-noise samples to use
        :return:
            [0]: Stay kernel (all kernels go HIST_SECONDS*FRAME_RATE into the past and FRAME_RATE into the future)
            [1]: Straight kernel
            [2]: Left kernel
            [3]: Right kernel
        """
        def kernel(t):
            indices = np.arange(n_samples)[btype_trace == t]
            ixm = indexing_matrix(indices, GlobalDefs.hist_seconds*GlobalDefs.frame_rate, GlobalDefs.frame_rate,
                                  int(n_samples))[0]
            return np.mean(stim[ixm]-np.mean(stim), 0)
        if self.switch_mean <= 0:
            stim = np.random.randn(int(n_samples)) * self.stim_std + self.stim_mean
        else:
            stim = np.zeros(int(n_samples))
            counter = 0
            switch_count = 0
            last_val = np.random.randn() * self.stim_std + self.stim_mean
            while counter < stim.size:
                if switch_count <= 0:
                    last_val = np.random.randn() * self.stim_std + self.stim_mean
                    switch_count = -1
                    while switch_count < 0:
                        switch_count = np.random.randn() * self.switch_std + self.switch_mean
                else:
                    switch_count -= 1
                stim[counter] = last_val
                counter += 1
        btype_trace = self.compute_openloop_behavior(stim)[0]
        return kernel(0), kernel(1), kernel(2), kernel(3)


class BoutFrequencyEvolver(CircleGradSimulation):
    """
    Class to obtain parameters that turn the output of the temperature branch of a ZfGpNetworkModel into bout frequency
    such that gradient navigation efficiency will be maximised by running an evolutionary algorithm
    """
    def __init__(self, stds: GradientStandards, model: ZfGpNetworkModel, n_sel_best=10, n_sel_random=6, n_progeny=2):
        """
        Creates a new BoutFrequencyEvolver
        :param stds: Data standardizations
        :param model: The network model to use
        :param n_sel_best: The number of top performing networks to select for each next generation
        :param n_sel_random: The number of not top performing networks to select for each next generation
        :param n_progeny: For each network pairing the number of child networks to produce
        """
        super().__init__(model, stds, 100, 22, 37, 26)
        self.model_branch = 't'
        self.n_sel_best = n_sel_best
        self.n_sel_random = n_sel_random
        self.n_progeny = n_progeny
        self.weight_mat = np.random.randn(self.n_networks, self.n_weights)
        self.generation = 0

    def run_ideal(self, nsteps, pfail=0.0):
        raise NotImplementedError("Function removed in this class")

    def run_simulation(self, nsteps, debug=False):
        """
        Runs simulation across all networks in a quasi-parallel manner
        :param nsteps: The number of simulation steps to perform
        :param debug: IGNORED
        :return: Returns a list of position arrays for each network
        """
        history = GlobalDefs.frame_rate * GlobalDefs.hist_seconds
        burn_period = history * 2
        start = history + 1
        net_pos = []
        for i in range(self.n_networks):
            p = np.full((nsteps + burn_period, 3), np.nan)
            p[:start + 1, :] = self.get_start_pos()[None, :]
            net_pos.append(p)
        steps = np.full(self.n_networks, start)  # for each simulation its current step
        # to avoid network queries we only evaluate bout frequencies of networks every ~ 20 steps
        last_bf_eval = np.full_like(steps, -100)  # for each network the last step in which bf was evaluated
        bf_eval_model_in = np.zeros((self.n_networks, 3, history, 1))  # model input to evaluate bfreq across networks
        bfreqs = np.full(self.n_networks, self.p_move)  # initialize all bout frequencies to base value
        while np.all(steps < nsteps + burn_period):  # run until first network is finished
            if np.all(steps-last_bf_eval >= 20) or np.any(steps-last_bf_eval >= 50):
                # newly evaluate bout frequencies
                for i in range(self.n_networks):
                    t = self.temperature(net_pos[i][steps[i] - history:steps[i], 0],
                                         net_pos[i][steps[i] - history:steps[i], 1])
                    bf_eval_model_in[i, 0, :, 0] = t
                bf_eval_model_in -= self.temp_mean  # these operations are ok since we only care about the
                bf_eval_model_in /= self.temp_std  # temperature input part - rest can be arbitrary values
                branch_out = self.model.branch_output(self.model_branch, bf_eval_model_in, self.remove)
                bfreqs = np.sum(branch_out * self.weight_mat, 1)
                # apply non-linearity
                bfreqs = 1 / (1 + np.exp(-bfreqs))  # [0, 1]
                bfreqs = (2 - 0.5) * bfreqs + 0.5  # [0.5, 2]
                bfreqs /= GlobalDefs.frame_rate  # turn into probability
                last_bf_eval = steps.copy()  # update indicator
            # determine which networks should move in this step - one decider draw for all
            d = self._uni_cash.next_rand()
            non_movers = np.nonzero(d > bfreqs)[0]
            movers = np.nonzero(d <= bfreqs)[0]
            for ix in non_movers:
                s = steps[ix]
                net_pos[ix][s, :] = net_pos[ix][s - 1, :]
                steps[ix] = s+1
            if movers.size == 0:
                continue
            # for each mover compute model prediction
            for n, ix in enumerate(movers):
                s = steps[ix]
                t = self.temperature(net_pos[ix][s - history:s, 0], net_pos[ix][s - history:s, 1])
                bf_eval_model_in[n, 0, :, 0] = (t - self.temp_mean) / self.temp_std
                spd = np.sqrt(np.sum(np.diff(net_pos[ix][s - history - 1:s, 0:2], axis=0) ** 2, 1))
                bf_eval_model_in[n, 1, :, 0] = (spd - self.disp_mean) / self.disp_std
                dang = np.diff(net_pos[ix][s - history - 1:s, 2], axis=0)
                bf_eval_model_in[n, 2, :, 0] = (dang - self.ang_mean) / self.ang_std
            model_out = self.model.predict(bf_eval_model_in[:movers.size, :, :, :], 1.0, self.remove)
            # for each mover turn model prediction into behavior and execute
            for n, ix in enumerate(movers):
                s = steps[ix]
                proj_diff = np.abs(model_out[n, :] - (self.t_preferred - self.temp_mean) / self.temp_std)
                behav_ranks = np.argsort(proj_diff)
                bt = self.select_behavior(behav_ranks)
                if bt == "N":
                    net_pos[ix][s, :] = net_pos[ix][s - 1, :]
                    steps[ix] = s+1
                    continue
                traj = self.get_bout_trajectory(net_pos[ix][s - 1, :], bt)
                if s + self.blen <= nsteps + burn_period:
                    net_pos[ix][s:s+self.blen, :] = traj
                else:
                    net_pos[ix][s:, :] = traj[:net_pos[ix][s:, :].shape[0], :]
                steps[ix] = s + self.blen
        return [pos[burn_period:steps[i], :] for i, pos in enumerate(net_pos)]

    def score_networks(self, nsteps):
        """
        For each network runs simulation and rates the error as the average temperature distance
        from the desired temperature weighted by the inverse of the radius
        :param nsteps: The number of simulation steps to (approximately) perform for each network
        :return:
            [0]: Average deviation from desired temperature for each network
            [1]: List of position traces of each network
        """
        net_pos = self.run_simulation(nsteps)
        error_scores = np.full(self.n_networks, np.nan)
        for i, pos in enumerate(net_pos):
            temperatures = self.temperature(pos[:, 0], pos[:, 1])
            weights = 1 / np.sqrt(np.sum(pos[:, :2]**2, 1))
            weights[np.isinf(weights)] = 0  # this only occurs when the "fish" is exactly at 0,0 at start, so remove
            sum_of_weights = np.nansum(weights)
            weighted_sum = np.nansum(np.abs(temperatures - self.t_preferred) * weights)
            assert not np.isnan(weighted_sum)
            assert not np.isnan(sum_of_weights)
            error_scores[i] = weighted_sum / sum_of_weights
        return error_scores, net_pos

    def next_generation(self, sel_index: np.ndarray, mut_std=0.1):
        """
        Advances the network generation using the networks identified by sel_index as parents and updates weights
        :param sel_index: The networks to be used as parents of next generation
        :param mut_std: The standard deviation of mutation noise (if <=0 no mutation is performed)
        :return: The new weights for convenience
        """
        if sel_index.size != (self.n_sel_best+self.n_sel_random):
            raise ValueError("The number of selected indices has to be {0}".format(self.n_sel_random+self.n_sel_best))
        new_weight_mat = np.full((self.n_networks, self.n_weights), np.nan)
        counter = 0
        for i1 in sel_index:
            for i2 in sel_index:
                for ic in range(self.n_progeny):
                    if i1 != i2:
                        # randomly mix parents weights - CROSSOVER
                        selector = np.random.rand(self.n_weights)
                        new_weight_mat[counter, selector < 0.5] = self.weight_mat[i1, selector < 0.5]
                        new_weight_mat[counter, selector >= 0.5] = self.weight_mat[i2, selector >= 0.5]
                    else:
                        new_weight_mat[counter, :] = self.weight_mat[i1, :]
                    # MUTATE
                    if mut_std > 0:
                        new_weight_mat[counter, :] += np.random.randn(self.n_weights) * mut_std
                    counter += 1
        assert not np.any(np.isnan(new_weight_mat))
        self.weight_mat = new_weight_mat
        self.generation += 1
        return self.weight_mat

    def evolve(self, n_eval_steps, mut_std=0.1):
        """
        Performs one full evolutionary step - scoring all current network and creating the next generation
        :param n_eval_steps: The number of steps to use for network scoring simulation
        :param mut_std: The standard deviation of mutation noise (if <=0 no mutation is performed)
        :return:
            [0]: The current errors
            [1]: The networ positions during simulation
            [2]: The updated weights
        """
        errors, net_pos = self.score_networks(n_eval_steps)
        ranks = np.argsort(errors)
        top = ranks[:self.n_sel_best]
        other = np.random.choice(ranks[self.n_sel_best:], self.n_sel_random, replace=False)
        return errors, net_pos, self.next_generation(np.r_[top, other], mut_std)

    @property
    def n_networks(self):
        """
        The total number of networks in each generation
        """
        return (self.n_sel_best+self.n_sel_random)**2 * self.n_progeny

    @property
    def n_weights(self):
        """
        The number of weights in each network
        """
        return self.model.n_units[0]
