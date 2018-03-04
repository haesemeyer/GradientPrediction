#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Classes for C. elegans behavior simulations
Behavior is modeled after data in (Ryu, 2002)
Note that behavior is merely approximated as the goal is not to derive a realistic simulation of C elegans behavior
"""

import numpy as np
from core import FRAME_RATE, HIST_SECONDS
from core import RandCash, GradientData, GradientStandards, CeGpNetworkModel, indexing_matrix


PRED_WINDOW = int(FRAME_RATE * 60)  # the model should predict the temperature 1 minute into the future

# Run-speed: 16 +/- 2 mm/min
# Model as appropriately per-step scaled gaussian: ~ N(16 mm/min, 2 mm/min)
# => Run-speed ~ N(16/(60*FRAME_RATE), 2/np.sqrt(60*FRAME_RATE))

# Run-durations: Exponentially distributed. avg=14s up gradient, avg=25s down gradient
# To roughly support these durations evaluate our model with a probability that matches 1/10 Hz (so that if reversal and
# sharp turn are the two top behaviors we will get a termination about once in 14 seconds. If they are bottom two,
# runs will only be terminated once in 50 s however
# => p_eval = 0.1/FRAME_RATE

# To introduce curvature into runs, add scaled angle at every step ~ N(0, 1deg/s)
# => jitter ~ N(0, .1/np.sqrt(FRAME_RATE))

# Runs terminated by undirectional sharp turns (pirouettes) or reversals
# Model sharp turns as gaussian ~ N(45 degrees, 5 degrees), smoothly executed as run bias over one second
# Model reversals as gaussian ~ N(180 degrees, 10 degrees), smoothly executed as run bias over one second

# To potentially allow for isothermal tracking introduce two shallow *directional* turns
# Small-turn right modeled as N ~ (10 deg, 1 deg), smoothly executed as run bias over one second
# Small-turn left modeled as N ~ (-10 deg, 1 deg), smoothly executed as run bias over one second

# => Behaviors: Continue-run; Sharp turn (random dir); Reverse; Small turn left; Small turn right
# Rank evaluations: Top->50%; 2nd->20%; 3rd-5th->10%


class TemperatureArena:
    """
    Base class for behavioral simulations that happen in a
    virtual arena where temperature depends on position
    """
    def __init__(self):
        # Set bout parameters used in the simulation
        self.p_move = 0.1 / FRAME_RATE  # Pick action on average once every 10s
        self.alen = int(FRAME_RATE)  # Actions happen over 1 s length
        # Per-frame displacement is drawn from normal distribution
        self.mu_disp = 16 / (60*FRAME_RATE)
        self.sd_disp = 5.3 / (60*FRAME_RATE)
        # Sharp turn, pirouette and shallow turn angles are drawn from gaussians
        self.mu_sharp = np.deg2rad(45)
        self.sd_sharp = np.deg2rad(5)
        self.mu_pir = np.deg2rad(180)
        self.sd_pir = np.deg2rad(10)
        self.mu_shallow = np.deg2rad(10)
        self.sd_shallow = np.deg2rad(1)
        # Per-frame crawl jitter is drawn from gaussian distribution
        self.mu_jitter = 0
        self.sd_jitter = np.deg2rad(.1 / np.sqrt(FRAME_RATE))
        # set up cashes of random numbers for movement parameters
        self._disp_cash = RandCash(1000, lambda s: np.abs(np.random.randn(s) * self.sd_disp + self.mu_disp))
        self._sharp_cash = RandCash(1000, lambda s: np.random.randn(s) * self.sd_sharp + self.mu_sharp)
        self._pir_cash = RandCash(1000, lambda s: np.random.randn(s) * self.sd_pir + self.mu_pir)
        self._shallow_cash = RandCash(1000, lambda s: np.random.randn(s) * self.sd_shallow + self.mu_shallow)
        self._jitter_cash = RandCash(1000, lambda s: np.random.randn(s) * self.sd_jitter + self.mu_jitter)
        # set up cash for left-right and movement deciders
        self._decider = RandCash(1000, lambda s: np.random.rand(s))
        # place holder to receive action trajectories for efficiency
        self._action = np.empty((self.alen, 3), np.float32)
        self._pos_cache = np.empty((1, 3), np.float32)

    def temperature(self, x, y):
        """
        Returns the temperature at the given positions
        """
        raise NotImplementedError("ABSTRACT")

    def arena_center(self):
        """
        Returns the x and y coordinates of the arena center
        """
        raise NotImplementedError("ABSTRACT")

    def out_of_bounds(self, x, y):
        """
        Detects whether the given x-y position is out of the arena
        :param x: The x position
        :param y: The y position
        :return: True if the given position is outside the arena, false otherwise
        """
        raise NotImplementedError("ABSTRACT")

    def adjust_oob_heading(self, p: np.ndarray) -> float:
        """
        Returns a heading given a 3-element position vector that will make the worm face the arena center
        with minimal angular displacement
        :param p: The out-of-bounds position
        :return: New suggested heading
        """
        center_x, center_y = self.arena_center()
        # adjust current heading such that worm faces center of arena - but update such that we don't
        # totally reset the already accumulated heading angle
        dx = p[0] - center_x
        dy = p[1] - center_y
        a = np.arctan2(dy, dx)
        curr_ang = p[2]
        new_heading = np.arctan2(np.sin(a - np.pi), np.cos(a - np.pi))
        # if we are already facing roughly in the same direction don't do anything - only act
        # if we are facing away from center by at least 0.15 radians
        d_angle = np.abs(new_heading - np.arctan2(np.sin(curr_ang), np.cos(curr_ang)))
        if d_angle > 0.15:
            tau_count = curr_ang // (2 * np.pi)
            return new_heading + tau_count * 2 * np.pi
        else:
            return curr_ang

    def get_action_trajectory(self, start, action_type="S", expected=False):
        """
        Gets a trajectory for the given bout type
        :param start: Tuple/vector of x, y, angle at start of turning action
        :param action_type: The type of action: (S)harp turn, (P)irouette, (L)eft small turn, (R)ight small turn
        :param expected: If true, instead of picking random angle pick expected value
        :return: The trajectory of the action (alen rows, 3 columns: x, y, angle)
        """
        if action_type not in ["S", "P", "L", "R"]:
            raise ValueError("action_type has to be one of S, P, L or R")
        if action_type == "S":
            turn_mult = 1 if self._decider.next_rand() < 0.5 else -1
            if expected:
                da = turn_mult*self.mu_sharp
            else:
                da = turn_mult*self._sharp_cash.next_rand()
        elif action_type == "P":
            turn_mult = 1 if self._decider.next_rand() < 0.5 else -1
            if expected:
                da = turn_mult*self.mu_pir
            else:
                da = turn_mult*self._pir_cash.next_rand()
        elif action_type == "L":
            if expected:
                da = -self.mu_shallow
            else:
                da = -self._shallow_cash.next_rand()
        else:
            if expected:
                da = self.mu_shallow
            else:
                da = self._shallow_cash.next_rand()
        da /= self.alen
        cx, cy = start[0], start[1]
        for i in range(self.alen):
            disp = self.mu_disp if expected else self._disp_cash.next_rand()
            a = start[2] + da * (i+1)
            cx += np.cos(a) * disp
            cy += np.sin(a) * disp
            self._action[i, 0] = cx
            self._action[i, 1] = cy
            self._action[i, 2] = a
        return self._action

    def get_action_type(self):
        """
        With 1/4 probability for each type returns a random action type
        """
        dec = self._decider.next_rand()
        if dec < 0.25:
            return "S"
        elif dec < 0.5:
            return "P"
        elif dec < 0.75:
            return "L"
        else:
            return "R"

    def sim_forward(self, nsteps, start_pos, start_type):
        """
        Simulates a number of steps ahead
        :param nsteps: The number of steps to perform
        :param start_pos: The current starting conditions [x,y,a]
        :param start_type: The behavior to perform on the first step "C", "S", "P", "L", "R"
        :return: The position at each timepoint nsteps*[x,y,a]
        """
        if start_type not in ["C", "S", "P", "L", "R"]:
            raise ValueError("start_type has to be either (C)ontinue, (S)harp turn, (P)irouette, (L)eft or (R)right")
        if self._pos_cache.shape[0] != nsteps:
            self._pos_cache = np.zeros((nsteps, 3))
        all_pos = self._pos_cache
        if start_type == "C":
            cx, cy, ca = start_pos
            cx += self.mu_disp * np.cos(ca)
            cy += self.mu_disp * np.sin(ca)
            # do not jitter heading for start bout which follows expected trajectory rather than random
            all_pos[0, 0] = cx
            all_pos[0, 1] = cy
            all_pos[0, 2] = ca
            i = 1
        else:
            # if a start action should be drawn, draw the "expected" bout not a random one
            traj = self.get_action_trajectory(start_pos, start_type, True)
            if traj.size <= nsteps:
                all_pos[:traj.shape[0], :] = traj
                i = traj.size
            else:
                return traj[:nsteps, :]
        while i < nsteps:
            # check for out of bounds and re-orient if necessary
            if self.out_of_bounds(all_pos[i - 1, 0], all_pos[i - 1, 1]):
                all_pos[i - 1, 2] = self.adjust_oob_heading(all_pos[i - 1, :])
            dec = self._decider.next_rand()
            if dec < self.p_move:
                at = self.get_action_type()
                traj = self.get_action_trajectory(all_pos[i - 1, :], at)
                if i + self.alen <= nsteps:
                    all_pos[i:i + self.alen, :] = traj
                else:
                    all_pos[i:, :] = traj[:all_pos[i:, :].shape[0], :]
                i += self.alen
            else:
                cx, cy, ca = all_pos[i - 1, :]
                disp = self._disp_cash.next_rand()
                cx += disp * np.cos(ca)
                cy += disp * np.sin(ca)
                ca += self._jitter_cash.next_rand()
                all_pos[i, 0] = cx
                all_pos[i, 1] = cy
                all_pos[i, 2] = ca
                i += 1
        return all_pos


class TrainingSimulation(TemperatureArena):
    """
    Base class for simulations that generate network training data
    """
    def __init__(self):
        super().__init__()

    def temperature(self, x, y):
        """
        Returns the temperature at the given positions
        """
        raise NotImplementedError("ABSTRACT")

    def arena_center(self):
        """
        Returns the x and y coordinates of the arena center
        """
        raise NotImplementedError("ABSTRACT")

    def out_of_bounds(self, x, y):
        """
        Detects whether the given x-y position is out of the arena
        :param x: The x position
        :param y: The y position
        :return: True if the given position is outside the arena, false otherwise
        """
        raise NotImplementedError("ABSTRACT")

    def run_simulation(self, nsteps):
        """
        Forward run of random gradient exploration
        :param nsteps: The number of steps to simulate
        :return: The position and heading in the gradient at each timepoint
        """
        return self.sim_forward(nsteps, np.zeros(3), "C").copy()

    def create_dataset(self, sim_pos):
        """
        Creates a GradientData object by executing all behavioral choices at simulated positions once per second
        :param sim_pos: Previously created simulation trajectory
        :return: GradientData object with all necessary training in- and outputs
        """
        if sim_pos.shape[1] != 3:
            raise ValueError("sim_pos has to be nx3 array with xpos, ypos and heading at each timepoint")
        history = FRAME_RATE * HIST_SECONDS
        start = history + 1  # start data creation with enough history present
        btypes = ["C", "S", "P", "L", "R"]
        # create vector that selects one position every second on average
        sel = np.random.rand(sim_pos.shape[0]) < (1/FRAME_RATE)
        sel[:start+1] = False
        # initialize model inputs and outputs
        inputs = np.zeros((sel.sum(), 3, history), np.float32)
        outputs = np.zeros((sel.sum(), 5), np.float32)
        # loop over each position, simulating PRED_WINDOW into future to obtain real finish temperature
        curr_sel = 0
        for step in range(start, sim_pos.shape[0]):
            if not sel[step]:
                continue
            # obtain inputs at given step
            inputs[curr_sel, 0, :] = self.temperature(sim_pos[step-history+1:step+1, 0],
                                                      sim_pos[step-history+1:step+1, 1])
            spd = np.sqrt(np.sum(np.diff(sim_pos[step-history:step+1, 0:2], axis=0)**2, 1))
            inputs[curr_sel, 1, :] = spd
            inputs[curr_sel, 2, :] = np.diff(sim_pos[step-history:step+1, 2], axis=0)
            # select each possible behavior in turn starting from this step and simulate
            # the behavior into the future and obtain final temperature as output by
            # approximating PRED_WINDOW steps into the future moving in a straight line
            for i, b in enumerate(btypes):
                fpos = self.sim_forward(self.alen, sim_pos[step, :], b)[-1, :]
                fpos[0] += np.cos(fpos[2]) * PRED_WINDOW * self.mu_disp
                fpos[1] += np.sin(fpos[2]) * PRED_WINDOW * self.mu_disp
                outputs[curr_sel, i] = self.temperature(fpos[0], fpos[1])
            curr_sel += 1
        assert not np.any(np.sum(inputs, (1, 2)) == 0)
        assert not np.any(np.sum(outputs, 1) == 0)
        return GradientData(inputs, outputs, PRED_WINDOW)


class ModelSimulation(TemperatureArena):
    """
    Base class for simulations that use trained networks
    to perform gradient navigation
    """
    def __init__(self, model: CeGpNetworkModel, tdata, t_preferred):
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
        self.btypes = ["C", "S", "P", "L", "R"]
        # all starting positions have to be within bounds but x and y coordinates are further limted to +/- maxstart
        self.maxstart = 10
        # optionally holds a list of vectors to suppress activation in units that should be "ablated"
        self.remove = None

    def temperature(self, x, y):
        """
        Returns the temperature at the given positions
        """
        raise NotImplementedError("ABSTRACT")

    def arena_center(self):
        """
        Returns the x and y coordinates of the arena center
        """
        raise NotImplementedError("ABSTRACT")

    def out_of_bounds(self, x, y):
        """
        Detects whether the given x-y position is out of the arena
        :param x: The x position
        :param y: The y position
        :return: True if the given position is outside the arena, false otherwise
        """
        raise NotImplementedError("ABSTRACT")

    @property
    def max_pos(self):
        raise NotImplementedError("ABSTRACT")

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
        Given a ranking of choices returns the action type identifier to perform
        """
        decider = self._decider.next_rand()
        if decider < 0.5:
            return self.btypes[ranks[0]]
        elif decider < 0.7:
            return self.btypes[ranks[1]]
        elif decider < 0.8:
            return self.btypes[ranks[2]]
        elif decider < 0.9:
            return self.btypes[ranks[3]]
        else:
            return self.btypes[ranks[4]]

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
        t_out = np.zeros(5)  # for debug purposes to run true outcome simulations forward
        if debug:
            # debug dict only contains information for timesteps which were select for possible movement!
            debug_dict["curr_temp"] = np.full(nsteps, np.nan)  # the current temperature at this position
            debug_dict["pred_temp"] = np.full((nsteps, 4), np.nan)  # the network predicted temperature for each move
            debug_dict["sel_behav"] = np.zeros(nsteps, dtype="U1")  # the actually selected move
            debug_dict["true_temp"] = np.full((nsteps, 4), np.nan)  # the temperature if each move is simulated
        history = FRAME_RATE * HIST_SECONDS
        burn_period = history * 2
        start = history + 1
        pos = np.full((nsteps + burn_period, 3), np.nan)
        pos[:start + 1, :] = self.get_start_pos()[None, :]
        # run simulation
        step = start
        model_in = np.zeros((1, 3, history, 1))
        p_eval = self.p_move
        while step < nsteps + burn_period:
            # check for out of bounds and re-orient if necessary
            if self.out_of_bounds(pos[step - 1, 0], pos[step - 1, 1]):
                pos[step - 1, 2] = self.adjust_oob_heading(pos[step - 1, :])
            if self._decider.next_rand() > p_eval:
                cx, cy, ca = pos[step - 1, :]
                disp = self._disp_cash.next_rand()
                cx += disp * np.cos(ca)
                cy += disp * np.sin(ca)
                ca += self._jitter_cash.next_rand()
                pos[step, 0] = cx
                pos[step, 1] = cy
                pos[step, 2] = ca
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
            if bt == "C":
                cx, cy, ca = pos[step - 1, :]
                disp = self._disp_cash.next_rand()
                cx += disp * np.cos(ca)
                cy += disp * np.sin(ca)
                ca += self._jitter_cash.next_rand()
                pos[step, 0] = cx
                pos[step, 1] = cy
                pos[step, 2] = ca
                step += 1
                continue
            traj = self.get_action_trajectory(pos[step-1, :], bt)
            if step + self.alen <= nsteps + burn_period:
                pos[step:step + self.alen, :] = traj
            else:
                pos[step:, :] = traj[:pos[step:, :].shape[0], :]
            step += self.alen
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
        history = FRAME_RATE * HIST_SECONDS
        burn_period = history * 2
        start = history + 1
        pos = np.full((nsteps + burn_period, 3), np.nan)
        pos[:start + 1, :] = self.get_start_pos()[None, :]
        step = start
        # overall bout frequency at ~1 Hz
        p_eval = self.p_move
        t_out = np.zeros(5)
        while step < nsteps + burn_period:
            if self._decider.next_rand() > p_eval:
                cx, cy, ca = pos[step - 1, :]
                disp = self._disp_cash.next_rand()
                cx += disp * np.cos(ca)
                cy += disp * np.sin(ca)
                ca += self._jitter_cash.next_rand()
                pos[step, 0] = cx
                pos[step, 1] = cy
                pos[step, 2] = ca
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
            if self._decider.next_rand() < pfail:
                np.random.shuffle(behav_ranks)
            bt = self.select_behavior(behav_ranks)
            if bt == "C":
                cx, cy, ca = pos[step - 1, :]
                disp = self._disp_cash.next_rand()
                cx += disp * np.cos(ca)
                cy += disp * np.sin(ca)
                ca += self._jitter_cash.next_rand()
                pos[step, 0] = cx
                pos[step, 1] = cy
                pos[step, 2] = ca
                step += 1
                continue
            traj = self.get_action_trajectory(pos[step - 1, :], bt)
            if step + self.alen <= nsteps + burn_period:
                pos[step:step + self.alen, :] = traj
            else:
                pos[step:, :] = traj[:pos[step:, :].shape[0], :]
            step += self.alen
        return pos[burn_period:, :]


class CircleGradSimulation(ModelSimulation):
    """
    Implements a nn-Model based circular gradient navigation simulation
    """
    def __init__(self, model: CeGpNetworkModel, tdata, radius, t_min, t_max, t_preferred=None):
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

    def arena_center(self):
        return 0, 0

    @property
    def max_pos(self):
        return self.radius


class LinearGradientSimulation(ModelSimulation):
    """
    Implements a nn-Model based linear gradient navigation simulation
    """
    def __init__(self, model: CeGpNetworkModel, tdata, xmax, ymax, t_min, t_max, t_preferred=None):
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

    def arena_center(self):
        return self.xmax / 2, self.ymax / 2

    @property
    def max_pos(self):
        return self.xmax
