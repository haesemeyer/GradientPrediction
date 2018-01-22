#  Copyright 2017 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script to plot training progress and simulations and representations across previously
trained neural networks - this script is very data-set specific
"""

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
import matplotlib.pyplot as pl
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from core import CircleGradSimulation, LinearGradientSimulation
from core import ModelData, GradientData, GpNetworkModel, FRAME_RATE, ca_convolve, WhiteNoiseSimulation, PersistentStore
from analyzeTempResponses import trial_average, cluster_responses
import os
from pandas import DataFrame
import pickle
from hashlib import md5


# file definitions
base_path = "./model_data/Adam_1e-4/sepInput_mixTrain/"

paths_1024 = [f+'/' for f in os.listdir(base_path) if "_3m1024_" in f]
paths_512 = [f+'/' for f in os.listdir(base_path) if "_3m512_" in f]
paths_256 = [f+'/' for f in os.listdir(base_path) if "_3m256_" in f]
# simulation globals
n_steps = 2000000
TPREFERRED = 26
circle_sim_params = {"radius": 100, "t_min": 22, "t_max": 37, "t_preferred": TPREFERRED}
lin_sim_params = {"xmax": 100, "ymax": 100, "t_min": 22, "t_max": 37, "t_preferred": TPREFERRED}


class ModelStore(PersistentStore):
    """
    Hdf5 backed store of data derived from using network model
    """
    def __init__(self, db_file_name, read_only=False):
        super().__init__(db_file_name, read_only)

    @staticmethod
    def model_dir_name(model_path: str):
        """
        Takes a full model path and returns the name of the actual directory containing the model definitions
        :param model_path: The full model path
        :return: Name of the model directory
        """
        if '\\' in model_path:
            spl = model_path.split('\\')
        else:
            spl = model_path.split('/')
        if len(spl[-1]) > 0:
            return spl[-1]
        else:
            return spl[-2]


class SimulationStore(ModelStore):
    """
    Hdf5 backed store of simulation data
    """
    def __init__(self, db_file_name, read_only=False):
        super().__init__(db_file_name, read_only)

    @staticmethod
    def _val_ids(sim_type: str, network_state: str):
        """
        Validates id parameters
        :param sim_type: Indicator of simulation type
        :param network_state: Indicator of network state
        """
        val_types = ['r', 'l']
        if sim_type not in val_types:
            raise ValueError("sim_type {0} is not valid has to be one of {1}".format(sim_type, val_types))
        val_states = ['naive', 'trained', 'ideal', 'bfevolve']
        if network_state not in val_states:
            raise ValueError("network_state {0} is not valid has to be one of {1}".format(network_state, val_states))

    @staticmethod
    def _run_sim(model_path: str, sim_type: str, network_state: str, debug: bool, drop_list=None):
        """
        Run simulation to obtain position and possibly debug information
        :param model_path: Full path to the network model definitions
        :param sim_type: Indicator of simulation type
        :param network_state: Indicator of network state
        :param debug: If true return debug information as well
        :param drop_list: Optional det_drop dictionary of lists of units to keep or drop
        :return:
            [0]: n_steps x 3 matrix of position information at each step
            [1]: Debug dict if debug is true or None otherwise
        """
        global std
        global TPREFERRED
        global n_steps
        mdata = ModelData(model_path)
        if network_state == "naive":
            chk = mdata.FirstCheckpoint
        else:
            chk = mdata.LastCheckpoint
        gpn = GpNetworkModel()
        gpn.load(mdata.ModelDefinition, chk)
        if sim_type == "r":
            sim = CircleGradSimulation(gpn, std, **circle_sim_params)
        else:
            sim = LinearGradientSimulation(gpn, std, **lin_sim_params)
        sim.remove = drop_list
        if network_state == "bfevolve":
            ev_path = model_path + '/evolve/generation_weights.npy'
            weights = np.load(ev_path)
            w = np.mean(weights[-1, :, :], 0)
            sim.bf_weights = w
        if network_state == "ideal":
            return sim.run_ideal(n_steps)
        else:
            return sim.run_simulation(n_steps, debug)

    def get_sim_pos(self, model_path: str, sim_type: str, network_state: str, drop_list=None) -> np.ndarray:
        """
        Retrieves simulation positions from the storage or runs simulation
        :param model_path: The full path to the network model
        :param sim_type: The simulation type, r = radial, l = linear
        :param network_state: The state of the netowrk: naive, trained, ideal or bfevolve
        :param drop_list: Optional det_drop dictionary of lists of units to keep or drop
        :return: The simulation positions
        """
        self._val_ids(sim_type, network_state)
        if drop_list is not None:
            # currently we do not store simulations that drop units
            return self._run_sim(model_path, sim_type, network_state, False, drop_list)
        mdir = self.model_dir_name(model_path)
        pos = self._get_data(mdir, sim_type, network_state, "pos")
        if pos is not None:
            return pos
        else:
            pos = self._run_sim(model_path, sim_type, network_state, False)
            self._set_data(pos, mdir, sim_type, network_state, "pos")
            return pos

    def get_sim_debug(self, model_path: str, sim_type: str, network_state: str, drop_list=None):
        """
        Retrieves simulation debug dict from the storage or runs simulation
        :param model_path: The full path to the network model
        :param sim_type: The simulation type, r = radial, l = linear
        :param network_state: The state of the netowrk: naive, trained, ideal or bfevolve
        :param drop_list: Optional det_drop dictionary of lists of units to keep or drop
        :return:
            [0]: The simulation positions
            [1]: The simulation debug dict
        """
        self._val_ids(sim_type, network_state)
        if network_state == "ideal":
            raise ValueError("debug information is currently not returned for ideal simulation")
        if drop_list is not None:
            # currently we do not store simulations that drop units
            return self._run_sim(model_path, sim_type, network_state, True, drop_list)
        mdir = self.model_dir_name(model_path)
        db_pickle = self._get_data(mdir, sim_type, network_state, "debug")
        pos = self._get_data(mdir, sim_type, network_state, "pos")
        if db_pickle is not None and pos is not None:
            deb_dict = pickle.loads(db_pickle)
            return pos, deb_dict
        else:
            pos, dbdict = self._run_sim(model_path, sim_type, network_state, True)
            self._set_data(pos, mdir, sim_type, network_state, "pos")
            self._set_data(np.void(pickle.dumps(dbdict)), mdir, sim_type, network_state, "debug")
            return pos, dbdict


class ActivityStore(ModelStore):
    """
    Hdf5 backed store of network cell activity data
    """
    def __init__(self, db_file_name, read_only=False):
        super().__init__(db_file_name, read_only)

    @staticmethod
    def _compute_cell_responses(model_dir, temp, network_id):
        """
        Loads a model and computes the temperature response of all neurons returning response matrix
        :param model_dir: The directory of the network model
        :param temp: The temperature input to test on the network
        :param network_id: Numerical id of the network to later relate units back to a network
        :return:
            [0]: n-timepoints x m-neurons matrix of responses
            [1]: 3 x m-neurons matrix with network_id in row 0, layer index in row 1, and unit index in row 2
        """
        global std
        mdata = ModelData(model_dir)
        gpn_trained = GpNetworkModel()
        gpn_trained.load(mdata.ModelDefinition, mdata.LastCheckpoint)
        # prepend lead-in to stimulus
        lead_in = np.full(gpn_trained.input_dims[2] - 1, np.mean(temp[:10]))
        temp = np.r_[lead_in, temp]
        act_dict = gpn_trained.unit_stimulus_responses(temp, None, None, std)
        if 't' in act_dict:
            activities = act_dict['t']
        else:
            activities = act_dict['m']
        activities = np.hstack(activities)
        # build id matrix
        id_mat = np.zeros((3, activities.shape[1]), dtype=np.int32)
        id_mat[0, :] = network_id
        if 't' in act_dict:
            hidden_sizes = [gpn_trained.n_units[0]] * gpn_trained.n_layers_branch
        else:
            hidden_sizes = [gpn_trained.n_units[1]] * gpn_trained.n_layers_mixed
        start = 0
        for layer, hs in enumerate(hidden_sizes):
            id_mat[1, start:start + hs] = layer
            id_mat[2, start:start + hs] = np.arange(hs, dtype=np.int32)
            start += hs
        return activities, id_mat

    def get_cell_responses(self, model_path: str, temperature: np.ndarray, network_id: int):
        """
        Obtain cell responses of all units in the given model for the given temperature stimulus
        :param model_path: The full path to the network model
        :param temperature: The temperature stimulus to present to the network
        :param network_id: An assigned numerical id of the network to later relate units back to a network
        :return:
            [0]: n-timepoints x m-neurons matrix of responses
            [1]: 3 x m-neurons matrix with network_id in row 0, layer index in row 1 and unit index in row 2
        """
        stim_hash = md5(temperature).hexdigest()
        mdir = self.model_dir_name(model_path)
        activities = self._get_data(mdir, stim_hash, "activities")
        id_mat = self._get_data(mdir, stim_hash, "id_mat")
        if activities is not None and id_mat is not None:
            # re-assign network id
            id_mat[0, :] = network_id
            return activities, id_mat
        activities, id_mat = self._compute_cell_responses(model_path, temperature, network_id)
        self._set_data(activities, mdir, stim_hash, "activities")
        self._set_data(id_mat, mdir, stim_hash, "id_mat")
        return activities, id_mat


def loss_file(path):
    return base_path + path + "losses.hdf5"


def mpath(path):
    return base_path + path[:-1]  # need to remove trailing slash


def train_loss(fname):
    lossfile = h5py.File(fname, "r")
    train_losses = np.array(lossfile["train_losses"])
    rank_errors = np.array(lossfile["train_rank_errors"])
    timepoints = np.array(lossfile["train_eval"])
    dfile.close()
    return timepoints, train_losses, rank_errors


def test_loss(fname):
    lossfile = h5py.File(fname, "r")
    test_losses = np.array(lossfile["test_losses"])
    rank_errors = np.array(lossfile["test_rank_errors"])
    timepoints = np.array(lossfile["test_eval"])
    return timepoints, test_losses, rank_errors


def plot_squared_losses():
    # assume timepoints same for all
    test_time = test_loss(loss_file(paths_512[0]))[0]
    test_256 = np.mean(np.vstack([test_loss(loss_file(lp))[1] for lp in paths_256]), 0)
    test_512 = np.mean(np.vstack([test_loss(loss_file(lp))[1] for lp in paths_512]), 0)
    test_1024 = np.mean(np.vstack([test_loss(loss_file(lp))[1] for lp in paths_1024]), 0)
    fig, ax = pl.subplots()
    ax.plot(test_time, np.log10(gaussian_filter1d(test_256, 2)), "C0.", label="256 HU")
    ax.plot(test_time, np.log10(gaussian_filter1d(test_512, 2)), "C1.", label="512 HU")
    ax.plot(test_time, np.log10(gaussian_filter1d(test_1024, 2)), "C2.", label="1024 HU")
    epoch_times = np.linspace(0, test_time.max(), 10, endpoint=False)
    for e in epoch_times:
        ax.plot([e, e], [-1.2, -0.5], 'k--', lw=0.5)
    ax.set_ylabel("log(Squared test error)")
    ax.set_xlabel("Training step")
    ax.legend()
    sns.despine()


def plot_rank_losses():
    # assume timepoints same for all
    test_time = test_loss(loss_file(paths_512[0]))[0]
    test_256 = np.mean(np.vstack([test_loss(loss_file(lp))[2] for lp in paths_256]), 0)
    test_512 = np.mean(np.vstack([test_loss(loss_file(lp))[2] for lp in paths_512]), 0)
    test_1024 = np.mean(np.vstack([test_loss(loss_file(lp))[2] for lp in paths_1024]), 0)
    fig, ax = pl.subplots()
    ax.plot(test_time, gaussian_filter1d(test_256, 2), "C0.", label="256 HU")
    ax.plot(test_time, gaussian_filter1d(test_512, 2), "C1.", label="512 HU")
    ax.plot(test_time, gaussian_filter1d(test_1024, 2), "C2.", label="1024 HU")
    epoch_times = np.linspace(0, test_time.max(), 10, endpoint=False)
    for e in epoch_times:
        ax.plot([e, e], [0, 5.1], 'k--', lw=0.5)
    ax.plot(test_time, np.full_like(test_time, 5), 'k--', label="Chance")
    ax.set_ylabel("Rank test error")
    ax.set_xlabel("Training step")
    ax.set_ylim(0, 5.1)
    ax.legend()
    sns.despine()


def bin_simulation(pos, bins: np.ndarray, simdir="r"):
    """
    Bin simulation results along the simulation direction, normalizigin occupancy in case of radial simulation
    :param pos: Position array obtained from running simulation
    :param bins: Array containing bin edges
    :param simdir: Determines whether occupancy should be calculated along (r)adius, (x)- or (y)-axis
    :return: Relative occupancy (corrected if radial)
    """
    if simdir not in ["r", "x", "y"]:
        raise ValueError("simdir has to be one of (r)adius, (x)- or (y)-axis")
    if simdir == "r":
        quantpos = np.sqrt(np.sum(pos[:, :2] ** 2, 1))
    elif simdir == "x":
        quantpos = pos[:, 0]
    else:
        quantpos = pos[:, 1]
    bin_centers = bins[:-1] + np.diff(bins) / 2
    h = np.histogram(quantpos, bins)[0].astype(float)
    # for radial histogram normalize by radius to offset area increase
    if simdir == "r":
        h = h / bin_centers
    h = h / h.sum()
    return h


def temp_convert(distances, sim_type):
    """
    Converts center or origin distances into temperature values according to our standard simulation types
    """
    if sim_type == "r":
        return distances / circle_sim_params["radius"] * (circle_sim_params["t_max"] - circle_sim_params["t_min"]) + \
               circle_sim_params["t_min"]
    else:
        return distances / lin_sim_params["radius"] * (lin_sim_params["t_max"] - lin_sim_params["t_min"]) + \
               lin_sim_params["t_min"]


def do_simulation(path, sim_type, run_ideal, drop_list=None):
    """
    Uses a model identified by path to run a naive and a trained and optionally an ideal and unit dropped simulation
    :param path: The model path
    :param sim_type: The simulation type to run
    :param run_ideal: If true, an ideal choice simulation will be run as well
    :param drop_list: If not none should be a list that will be fed to det_drop to determine which units are kept (1)
        or dropped (0)
    :return:
        [0]: The occupancy bin centers in degrees C
        [1]: The occupancy of the naive model
        [2]: The occupancy of the trained model
        [3]: The occupancy of the ideal choice model if run_ideal=True, None otherwise
        [4]: The occupancy of a unit dropped model if drop_list is provided, None otherwise
    """
    bins = np.linspace(0, circle_sim_params["radius"], 100)
    # bin-centers in degress
    bcenters = bins[:-1]+np.diff(bins)/2
    bcenters = temp_convert(bcenters, sim_type)
    if sim_type == "l":
        simdir = "x"
    else:
        simdir = "r"
    with SimulationStore("sim_store.hdf5") as sim_store:
        pos_naive = sim_store.get_sim_pos(path, sim_type, "naive")
        h_naive = bin_simulation(pos_naive, bins, simdir)
        pos_trained = sim_store.get_sim_pos(path, sim_type, "trained")
        h_trained = bin_simulation(pos_trained, bins, simdir)
        if run_ideal:
            pos_ideal = sim_store.get_sim_pos(path, sim_type, "ideal")
            h_ideal = bin_simulation(pos_ideal, bins, simdir)
        else:
            h_ideal = None
        if drop_list is not None:
            pos_drop = sim_store.get_sim_pos(path, sim_type, "trained", drop_list)
            h_drop = bin_simulation(pos_drop, bins, simdir)
        else:
            h_drop = None
    return bcenters, h_naive, h_trained, h_ideal, h_drop


def plot_sim(sim_type):
    all_n = []
    t_256 = []
    t_512 = []
    t_1024 = []
    bins = None
    for p256 in paths_256:
        bins, naive, trained = do_simulation(mpath(p256), sim_type, False)[:3]
        all_n.append(naive)
        t_256.append(trained)
    t_256 = np.vstack(t_256)
    for p512 in paths_512:
        _, naive, trained = do_simulation(mpath(p512), sim_type, False)[:3]
        all_n.append(naive)
        t_512.append(trained)
    t_512 = np.vstack(t_512)
    for p1024 in paths_1024:
        _, naive, trained = do_simulation(mpath(p1024), sim_type, False)[:3]
        all_n.append(naive)
        t_1024.append(trained)
    t_1024 = np.vstack(t_1024)
    all_n = np.vstack(all_n)
    fig, ax = pl.subplots()
    sns.tsplot(t_256, bins, n_boot=1000, ax=ax, color="C0")
    ax.plot(bins, np.mean(t_256, 0), lw=2, label="256 HU", c="C0")
    sns.tsplot(t_512, bins, n_boot=1000, ax=ax, color="C1")
    ax.plot(bins, np.mean(t_512, 0), lw=2, label="512 HU", c="C1")
    sns.tsplot(t_1024, bins, n_boot=1000, ax=ax, color="C2")
    ax.plot(bins, np.mean(t_1024, 0), lw=2, label="1024 HU", c="C2")
    sns.tsplot(all_n, bins, n_boot=1000, ax=ax, color="k")
    ax.plot(bins, np.mean(all_n, 0), "k", lw=2, label="Naive")
    ax.plot([TPREFERRED, TPREFERRED], ax.get_ylim(), 'C4--')
    ax.set_ylim(0)
    ax.legend()
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Temperature")
    sns.despine(fig, ax)


def create_det_drop_list(network_id, cluster_ids, unit_ids, clust_to_drop, shuffle=False):
    """
    Creates a network specific list of deterministic unit drop vectors
    :param network_id: The id of the network for which to generate drop vectors
    :param cluster_ids: For each analyzed unit its cluster membership
    :param unit_ids: For each analyzed unit 3xneurons matrix with network_id, layer index and unit index
    :param clust_to_drop: Id or list of ids of cluster members to drop
    :param shuffle: If true drop indicator will be shuffled in each layer
    :return: List of deterministic drop vectors
    """
    if type(clust_to_drop) is not list:
        clust_to_drop = [clust_to_drop]
    # use unit_ids to identify topology of network and drop requested units
    det_drop = []
    for l_id in np.unique(unit_ids[1, unit_ids[0, :] == network_id]):
        in_network_layer = np.logical_and(unit_ids[0, :] == network_id, unit_ids[1, :] == l_id)
        drop = np.ones(in_network_layer.sum())
        for cd in clust_to_drop:
            drop[cluster_ids[in_network_layer] == cd] = 0
        if shuffle:
            np.random.shuffle(drop)
        det_drop.append(drop)
    return {'t': det_drop}


def plot_sim_debug(path, sim_type, drop_list=None):
    """
    Runs indicated simulation on fully trained network, retrieves debug information and plots parameter correlations
    :param path: The model path
    :param sim_type: Either "r"adial or "l"inear
    :param drop_list: Optional list of vectors that indicate which units should be kept (1) or dropped (0)
    :return:
        [0]: The simulation positions
        [1]: The debug dict
    """
    with SimulationStore("sim_store.hdf5") as sim_store:
        all_pos, db_dict = sim_store.get_sim_debug(path, sim_type, "trained", drop_list)
    ct = db_dict["curr_temp"]
    val = np.logical_not(np.isnan(ct))
    ct = ct[val]
    pred = db_dict["pred_temp"][val, :]
    selb = db_dict["sel_behav"][val]
    tru = db_dict["true_temp"][val, :]
    btypes = ["N", "S", "L", "R"]
    # plot counts of different behavior types
    fig, ax = pl.subplots()
    sns.countplot(selb, order=btypes)
    sns.despine(fig, ax)
    # for each behavior type, plot scatter of prediction vs. current temperature
    fig, axes = pl.subplots(2, 2)
    axes = axes.ravel()
    for i in range(4):
        axes[i].scatter(ct, pred[:, i], s=2)
        axes[i].set_xlabel("Current temperature")
        axes[i].set_ylabel("{0} prediction".format(btypes[i]))
        axes[i].set_title("r = {0:.2g}".format(np.corrcoef(ct, pred[:, i])[0, 1]))
    sns.despine(fig)
    fig.tight_layout()
    # for each behavior type, plot scatter of prediction vs.true outcome
    fig, axes = pl.subplots(2, 2)
    axes = axes.ravel()
    for i in range(4):
        axes[i].scatter(tru[:, i], pred[:, i], s=2)
        axes[i].set_xlabel("{0} tru outcome".format(btypes[i]))
        axes[i].set_ylabel("{0} prediction".format(btypes[i]))
        axes[i].set_title("r = {0:.2g}".format(np.corrcoef(tru[:, i], pred[:, i])[0, 1]))
    sns.despine(fig)
    fig.tight_layout()
    # Plot average rank errors binned by current temperature
    rerbins = 10
    avg_rank_errors = np.zeros(rerbins)
    ctb = np.linspace(ct.min(), ct.max(), rerbins+1)
    bincents = ctb[:-1] + np.diff(ctb)/2
    for i in range(rerbins):
        in_bin = np.logical_and(ct >= ctb[i], ct < ctb[i+1])
        pib = pred[in_bin, :]
        tib = tru[in_bin, :]
        errsum = 0
        for j in range(pib.shape[0]):
            p_ranks = np.unique(pib[j, :], return_inverse=True)[1]
            t_ranks = np.unique(tib[j, :], return_inverse=True)[1]
            errsum += np.sum(np.abs(p_ranks - t_ranks))
        avg_rank_errors[i] = errsum / pib.shape[0]
    fig, ax = pl.subplots()
    ax.plot(bincents, avg_rank_errors, 'o')
    ax.set_title("Avg. rank errors by temperature")
    ax.set_xlabel("Binned start temperature")
    ax.set_ylabel("Average rank error")
    sns.despine(fig, ax)
    return all_pos, db_dict


def plot_fish_nonfish_analysis(sim_type="r"):
    """
    Analyzes ablations of fish and non-fish clusters and plots
    """
    def sim_info(net_id):
        def bin_pos(all_pos):
            nonlocal sim_type
            nonlocal bins
            bin_centers = bins[:-1] + np.diff(bins) / 2
            if sim_type == "r":
                quantpos = np.sqrt(np.sum(all_pos[:, :2] ** 2, 1))
            else:
                quantpos = all_pos[:, 0]
            h = np.histogram(quantpos, bins)[0].astype(float)
            # normalize for radius if applicable
            if sim_type == "r":
                h /= bin_centers
            h /= h.sum()
            # convert bin_centers to temperature
            bin_centers = temp_convert(bin_centers, sim_type)
            return bin_centers, h

        def temp_error(all_pos):
            nonlocal sim_type
            if sim_type == "r":
                quantpos = np.sqrt(np.sum(all_pos[:, :2] ** 2, 1))
            else:
                quantpos = all_pos[:, 0]
            temp_pos = temp_convert(quantpos, sim_type)
            if sim_type == "r":
                # form a weighted average, considering points of larger radius less since just by
                # chance they will be visited more often
                weights = 1 / np.sqrt(np.sum(all_pos[:, :2]**2, 1))
                weights[np.isinf(weights)] = 0  # occurs when 0,0 was picked as starting point only
                sum_of_weights = np.nansum(weights)
                weighted_sum = np.nansum(np.sqrt((temp_pos - TPREFERRED)**2) * weights)
                return weighted_sum / sum_of_weights
            return np.mean(np.sqrt((temp_pos - TPREFERRED)**2))

        nonlocal sim_type
        nonlocal fish
        nonlocal non_fish
        fish_remove = create_det_drop_list(net_id, clust_ids, all_ids, fish)
        nonfish_remove = create_det_drop_list(net_id, clust_ids, all_ids, non_fish)
        shuff_remove = create_det_drop_list(net_id, clust_ids, all_ids, fish, True)
        with SimulationStore("sim_store.hdf5") as sim_store:
            pos_naive, db_naive = sim_store.get_sim_debug(mpath(paths_512[net_id]), sim_type, "naive")
            pos_trained, db_trained = sim_store.get_sim_debug(mpath(paths_512[net_id]), sim_type, "trained")
            pos_fish, db_fish = sim_store.get_sim_debug(mpath(paths_512[net_id]), sim_type, "trained", fish_remove)
            pos_nonfish, db_nonfish = sim_store.get_sim_debug(mpath(paths_512[net_id]), sim_type, "trained",
                                                              nonfish_remove)
            pos_shuff, db_shuff = sim_store.get_sim_debug(mpath(paths_512[net_id]), sim_type, "trained", shuff_remove)
        bins = np.linspace(0, circle_sim_params["radius"], 100)
        bc, h_naive = bin_pos(pos_naive)
        e_naive = temp_error(pos_naive)
        h_trained = bin_pos(pos_trained)[1]
        e_trained = temp_error(pos_trained)
        h_fish = bin_pos(pos_fish)[1]
        e_fish = temp_error(pos_fish)
        h_nonfish = bin_pos(pos_nonfish)[1]
        e_nonfish = temp_error(pos_nonfish)
        h_shuff = bin_pos(pos_shuff)[1]
        e_shuff = temp_error(pos_shuff)
        return bc, {"naive": (h_naive, db_naive, e_naive), "trained": (h_trained, db_trained, e_trained),
                    "fish": (h_fish, db_fish, e_fish), "nonfish": (h_nonfish, db_nonfish, e_nonfish),
                    "shuffle": (h_shuff, db_shuff, e_shuff)}

    def prediction_stats(db_dict):
        """
        Computes basic statistics on the quality of network predictions using linear regression between prediction
        and true values
        :param db_dict: The debug dictionary with simulation information
        :return:
            [0]: List of slopes (length four, one for each behavior)
            [1]: List of intercept
            [2]: List of correlations (r-values)
        """
        ct = db_dict["curr_temp"]
        val = np.logical_not(np.isnan(ct))
        pred = db_dict["pred_temp"][val, :]
        tru = db_dict["true_temp"][val, :]
        sl, ic, co = [], [], []
        for i in range(4):
            s, i, c = linregress(pred[:, i], tru[:, i])[:3]
            sl.append(s)
            ic.append(i)
            co.append(c)
        return [np.mean(sl)], [np.mean(ic)], [np.mean([co])]

    def rank_errors(db_dict, temp_bins):
        """
        Computes prediction rank errors binned by temperature
        :param db_dict: The debug dictionary with simulation information
        :param temp_bins: The temperature bin edges in which to evaluate the rank errors
        :return: temp_bins-1 long vector with the average rank errors in each bin
        """
        ct = db_dict["curr_temp"]
        val = np.logical_not(np.isnan(ct))
        pred = db_dict["pred_temp"][val, :]
        tru = db_dict["true_temp"][val, :]
        ct = ct[val]
        avg_rank_errors = np.zeros(temp_bins.size - 1)
        for i in range(temp_bins.size - 1):
            in_bin = np.logical_and(ct >= temp_bins[i], ct < temp_bins[i + 1])
            pib = pred[in_bin, :]
            tib = tru[in_bin, :]
            errsum = 0
            for j in range(pib.shape[0]):
                p_ranks = np.unique(pib[j, :], return_inverse=True)[1]
                t_ranks = np.unique(tib[j, :], return_inverse=True)[1]
                errsum += np.sum(np.abs(p_ranks - t_ranks))
            if pib.shape[0] > 0:
                avg_rank_errors[i] = errsum / pib.shape[0]
            else:
                avg_rank_errors[i] = np.nan
        return avg_rank_errors

    # get fish and non-fish clusters based on user input
    global n_regs
    all_clust = list(range(n_regs))
    fish = []
    failed = True
    while failed:
        try:
            fish = [int(x) for x in input("Input fish like cluster numbers separated by space: ").split()]
            failed = False
        except ValueError:
            print("Invalid input. Retry.", flush=True)
    non_fish = [elem for elem in all_clust if elem not in fish]
    print("Fish clusters: ", fish)
    print("Non-fish clusters: ", non_fish)
    colors = {"naive": "k", "trained": "C0", "fish": "C3", "nonfish": "C1", "shuffle": "C2"}
    labels = {"naive": "Naive", "trained": "Trained", "fish": "Fish removed", "nonfish": "NonFish removed",
              "shuffle": "Shuffled removal"}
    dists = None
    corrs = None
    slopes = None
    r_errors = None
    grad_errors = None
    tbins = np.linspace(22, 37, 40)
    for nid in range(len(paths_512)):
        print("Network id = ", nid)
        bcents, results = sim_info(nid)
        if corrs is None:
            corrs = {k: [] for k in results.keys()}
            slopes = {k: [] for k in results.keys()}
            dists = {k: [] for k in results.keys()}
            r_errors = {k: [] for k in results.keys()}
            grad_errors = {k: [] for k in results.keys()}
        for k in results.keys():
            s, _, c = prediction_stats(results[k][1])
            corrs[k] += c
            slopes[k] += s
            dists[k].append(results[k][0])
            r_errors[k].append(rank_errors(results[k][1], tbins))
            grad_errors[k].append(results[k][2])
    # plot gradient distributions of models
    fig, ax = pl.subplots()
    for k in dists.keys():
        sns.tsplot(dists[k], bcents, ax=ax, color=colors[k])
        ax.plot(bcents, np.mean(dists[k], 0), color=colors[k], label=labels[k])
    ax.set_ylim(0)
    ax.plot([TPREFERRED, TPREFERRED], ax.get_ylim(), 'k--')
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Occupancy")
    ax.legend()
    sns.despine(fig, ax)
    # plot gradient position errors of models
    fig, ax = pl.subplots()
    dframe = DataFrame(grad_errors)
    sns.boxplot(data=dframe, order=["naive", "trained", "shuffle", "fish", "nonfish"])
    ax.set_ylabel("Average gradient position error [C]")
    ax.set_ylim(0)
    sns.despine(fig, ax)
    # plot scatter of prediction quality
    fig, ax = pl.subplots()
    for k in dists.keys():
        ax.scatter(corrs[k], slopes[k], c=colors[k], label=labels[k], alpha=0.8)
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Slope")
    ax.legend()
    sns.despine(fig, ax)
    # plot temperature binned rank errors
    tbc = tbins[:-1] + np.diff(tbins) / 2
    fig, ax = pl.subplots()
    for k in r_errors.keys():
        sns.tsplot(r_errors[k], tbc, ax=ax, color=colors[k], estimator=np.nanmean)
        ax.plot(tbc, np.nanmean(r_errors[k], 0), color=colors[k], label=labels[k])
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Prediction rank error")
    ax.legend()
    sns.despine(fig, ax)
# def plot_fish_nonfish_analysis


def compute_gradient_bout_frequency(model_path, drop_list=None):
    def bout_freq(pos: np.ndarray):
        r = np.sqrt(np.sum(pos[:, :2]**2, 1))  # radial position
        spd = np.r_[0, np.sqrt(np.sum(np.diff(pos[:, :2], axis=0) ** 2, 1))]  # speed
        bs = np.r_[0, np.diff(spd) > 0.00098]  # bout starts
        bins = np.linspace(0, 100, 6)
        bcenters = bins[:-1] + np.diff(bins)/2
        cnt_r = np.histogram(r, bins)[0]
        cnt_r_bs = np.histogram(r[bs > 0.1], bins)[0]
        bfreq = cnt_r_bs / cnt_r * FRAME_RATE
        return bfreq, bcenters

    with SimulationStore("sim_store.hdf5") as sim_store:
        pos_fixed = sim_store.get_sim_pos(model_path, "r", "trained", drop_list)
        pos_var = sim_store.get_sim_pos(model_path, "r", "bfevolve", drop_list)
    bf_fixed, bc = bout_freq(pos_fixed)
    bf_var, bc = bout_freq(pos_var)
    return bc, bf_fixed, bf_var


if __name__ == "__main__":
    # plot training progress
    plot_squared_losses()
    plot_rank_losses()
    # load training data for scaling
    try:
        std = GradientData.load_standards("gd_training_data.hdf5")
    except IOError:
        print("No standards found attempting to load full training data")
        std = GradientData.load("gd_training_data.hdf5").standards
    # plot radial sim results
    plot_sim("r")
    # load and interpolate temperature stimulus
    dfile = h5py.File("stimFile.hdf5", 'r')
    tsin = np.array(dfile['sine_L_H_temp'])
    x = np.arange(tsin.size)  # stored at 20 Hz !
    xinterp = np.linspace(0, tsin.size, tsin.size * FRAME_RATE // 20)
    temperature = np.interp(xinterp, x, tsin)
    dfile.close()
    # for our 512 unit network extract all temperature responses and correponding IDs
    all_cells = []
    all_ids = []
    for i, d in enumerate(paths_512):
        with ActivityStore("activity_store.hdf5") as act_store:
            cell_res, ids = act_store.get_cell_responses(mpath(d), temperature, i)
        all_cells.append(cell_res)
        all_ids.append(ids)
    all_cells = np.hstack(all_cells)
    all_ids = np.hstack(all_ids)
    # convolve all activity with the MTA derived nuclear Gcamp6s calcium kernel
    # we want to put network activity "on same footing" as imaging data
    tau_on = 1.4  # seconds
    tau_on *= FRAME_RATE  # in frames
    tau_off = 2  # seconds
    tau_off *= FRAME_RATE  # in frames
    kframes = np.arange(10 * FRAME_RATE)  # 10 s long kernel
    kernel = 2**(-kframes / tau_off) * (1 - 2**(-kframes / tau_on))
    kernel = kernel / kernel.sum()
    # convolve with our kernel
    for i in range(all_cells.shape[1]):
        all_cells[:, i] = ca_convolve(all_cells[:, i], 0, 0, kernel)
    # perform spectral clustering
    n_regs = 8
    clust_ids, coords = cluster_responses(all_cells, n_regs)
    # trial average the "cells"
    all_cells = trial_average(all_cells, 3)

    # collect cluster-average activities
    cluster_acts = np.zeros((all_cells.shape[0], n_regs))
    for i in range(n_regs):
        cluster_acts[:, i] = np.mean(all_cells[:, clust_ids == i], 1)

    # plot spectral embedding and cluster average activity
    fig = pl.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(n_regs):
        ax.scatter(coords[clust_ids == i, 0], coords[clust_ids == i, 1], coords[clust_ids == i, 2], s=5)
    fig, (ax_on, ax_off) = pl.subplots(ncols=2)
    time = np.arange(all_cells.shape[0]) / FRAME_RATE
    for i in range(n_regs):
        act = cluster_acts[:, i]
        if np.corrcoef(act, temperature[:act.size])[0, 1] < 0:
            sns.tsplot(all_cells[:, clust_ids == i].T, time, color="C{0}".format(i), ax=ax_off)
        else:
            sns.tsplot(all_cells[:, clust_ids == i].T, time, color="C{0}".format(i), ax=ax_on)
    ax_off.set_xlabel("Time [s]")
    ax_off.set_ylabel("Cluster average activity")
    ax_on.set_xlabel("Time [s]")
    ax_on.set_ylabel("Cluster average activity")
    sns.despine()
    fig.tight_layout()

    # plot cluster sizes
    fig, ax = pl.subplots()
    sns.countplot(clust_ids[clust_ids > -1], ax=ax)
    ax.set_ylabel("Cluster size")
    ax.set_xlabel("Cluster number")
    sns.despine(fig, ax)

    # plot white noise analysis of networks
    behav_kernels = {}
    k_names = ["stay", "straight", "left", "right"]
    for p in paths_512:
        mdata_wn = ModelData(mpath(p))
        gpn_wn = GpNetworkModel()
        gpn_wn.load(mdata_wn.ModelDefinition, mdata_wn.LastCheckpoint)
        wna = WhiteNoiseSimulation(std, gpn_wn)
        kernels = wna.compute_behavior_kernels(10000000)
        for i, n in enumerate(k_names):
            if n in behav_kernels:
                behav_kernels[n].append(kernels[i])
            else:
                behav_kernels[n] = [kernels[i]]
    kernel_time = np.linspace(-4, 1, behav_kernels['straight'][0].size)
    for n in k_names:
        behav_kernels[n] = np.vstack(behav_kernels[n])
    fig, ax = pl.subplots()
    for i, n in enumerate(k_names):
        sns.tsplot(behav_kernels[n], kernel_time, n_boot=1000, color="C{0}".format(i), ax=ax)
        ax.plot(kernel_time, np.mean(behav_kernels[n], 0), color="C{0}".format(i), label=n)
    ax.set_ylabel("Filter kernel")
    ax.set_xlabel("Time around bout [s]")
    ax.legend()
    sns.despine(fig, ax)

