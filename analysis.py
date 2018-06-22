#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Module for data analysis classes and functions
"""

from mo_types import MoTypes
from global_defs import GlobalDefs
import numpy as np
from data_stores import SimulationStore, ActivityStore
from core import GradientStandards
import os
import h5py
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import SpectralClustering


class Analyzer:
    """
    Class to abstract analysis by model system
    """
    def __init__(self, mo: MoTypes, std: GradientStandards, sim_store_name, act_store_name):
        """
        Creates a new analyzer
        :param mo: Objecto to identify the model organism and it's network models
        :param std: Standardizations for model inputs
        :param sim_store_name: File name of persistent storage of simulation data (or None)
        :param act_store_name: File name of persistent storage of activity data (or None)
        """
        self.mo = mo
        self.std = std
        self.sim_store_name = sim_store_name
        self.act_store_name = act_store_name

    def run_simulation(self, path, sim_type, network_state, debug=False, drop_list=None):
        """
        Uses a model identified by path to run a naive and a trained and optionally an ideal and unit dropped simulation
        :param path: The model path
        :param sim_type: The simulation type to run ("r"adial or "l"inear)
        :param network_state: The state of the network ("naive", "trained", "ideal", "bfevolve")
        :param debug: If true, also return debug dictionary
        :param drop_list: If not none should be a list that will be fed to det_drop to determine which units are kept
                (1) or dropped (0)
        :return:
            [0]: Array of x,y,angle positions
            [1]: If debug=True dictionary of simulation debug information
        """
        with SimulationStore(self.sim_store_name, self.std, self.mo) as sim_store:
            if debug:
                return sim_store.get_sim_debug(path, sim_type, network_state, drop_list)
            else:
                return sim_store.get_sim_pos(path, sim_type, network_state, drop_list)

    def temperature_activity(self, path, temperature, network_id):
        """
        Uses a model identified by path and returns activity of all cells in the temperature branch
        :param path: The model path
        :param temperature: The temperature stiumulus
        :param network_id: The network id for constructing correct unit ids
        :return:
            [0]: n-timepoints x m-neurons matrix of responses
            [1]: 3 x m-neurons matrix with network_id in row 0, layer index in row 1, and unit index in row 2
        """
        with ActivityStore(self.act_store_name, self.std, self.mo) as act_store:
            return act_store.get_cell_responses(path, temperature, network_id)


def bin_simulation(pos, bins: np.ndarray, simdir="r"):
    """
    Bin simulation results along the simulation direction, normalizing occupancy in case of radial simulation
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


def bin_simulation_pt(pos, bins: np.ndarray):
    """
    Bin simulation result facing angles
    :param pos: Position array obtained from running simulation
    :param bins: Array containing bin edges
    :return: Relative occupancy
    """
    quantpos = MoTypes(False).pt_sim.facing_angle(pos[:, 0], pos[:, 1], pos[:, 2])
    # remap angles from -pi to pi
    quantpos[quantpos > np.pi] = quantpos[quantpos > np.pi] - 2*np.pi
    quantpos[quantpos < -np.pi] = quantpos[quantpos < -np.pi] + 2*np.pi
    h = np.histogram(quantpos, bins)[0].astype(float)
    h = h / h.sum()
    return h


def temp_convert(distances, sim_type):
    """
    Converts center or origin distances into temperature values according to our standard simulation types
    """
    circle_sim_params = GlobalDefs.circle_sim_params
    lin_sim_params = GlobalDefs.lin_sim_params
    if sim_type == "r":
        return distances / circle_sim_params["radius"] * (circle_sim_params["t_max"] - circle_sim_params["t_min"]) \
               + circle_sim_params["t_min"]
    else:
        return distances / lin_sim_params["radius"] * (lin_sim_params["t_max"] - lin_sim_params["t_min"]) \
               + lin_sim_params["t_min"]


def create_det_drop_list(network_id, cluster_ids, unit_ids, clust_to_drop, shuffle=False, branch='t'):
    """
    Creates a network specific list of deterministic unit drop vectors for a network branch
    :param network_id: The id of the network for which to generate drop vectors
    :param cluster_ids: For each analyzed unit its cluster membership
    :param unit_ids: For each analyzed unit 3xneurons matrix with network_id, layer index and unit index
    :param clust_to_drop: Id or list of ids of cluster members to drop
    :param shuffle: If true drop indicator will be shuffled in each layer
    :param branch: The dictionary identifier to give to the drop list
    :return: Dictionary with list of deterministic drop vectors
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
    return {branch: det_drop}


def quant_pos(all_pos: np.ndarray, sim_type: str):
    """
    Uses simulation type to compute and return the gradient position relevant for temperature calculation
    """
    if sim_type == "r":
        return np.sqrt(np.sum(all_pos[:, :2] ** 2, 1))
    else:
        # assume gradient runs in x-direction for linear simulation
        return all_pos[:, 0]


def temp_error(all_pos: np.ndarray, sim_type: str):
    """
    Compute the (radius corrected) average deviance in degrees from the preferred temperature in a simulation run
    :param all_pos: Position array of a simulation
    :param sim_type: The type of simulation ("r"adial or "l"inear)
    :return: The average deviation from preferred temperature
    """
    qp = quant_pos(all_pos, sim_type)
    temp_pos = temp_convert(qp, sim_type)
    if sim_type == "r":
        # form a weighted average, considering points of larger radius less since just by
        # chance they will be visited more often
        weights = 1 / qp
        weights[np.isinf(weights)] = 0  # occurs when 0,0 was picked as starting point only
        sum_of_weights = np.nansum(weights)
        weighted_sum = np.nansum(np.sqrt((temp_pos - GlobalDefs.tPreferred) ** 2) * weights)
        return weighted_sum / sum_of_weights
    return np.mean(np.sqrt((temp_pos - GlobalDefs.tPreferred) ** 2))


def preferred_fraction(all_pos: np.ndarray, sim_type: str, delta_t=1.0):
    """
    Calculates the (radius corrected) occupancy within delta_t degrees of the preferred temperature
    :param all_pos: Position array of a simulation
    :param sim_type: The type of simulation ("r"adial or "l"inear)
    :param delta_t: Fraction within +/- deltaT degrees will be computed
    :return: Fraction spent around preferred temperature
    """
    qp = quant_pos(all_pos, sim_type)
    temp_pos = temp_convert(qp, sim_type)
    weights = np.ones(qp.size) if sim_type == "l" else 1/qp
    weights[np.isinf(weights)] = 0
    sum_of_weights = np.nansum(weights)
    in_delta = np.logical_and(temp_pos > GlobalDefs.tPreferred-delta_t, temp_pos < GlobalDefs.tPreferred+delta_t)
    return np.nansum(weights[in_delta]) / sum_of_weights


def trial_average(mat, n_trials):
    """
    Computes the trial average of each trace in mat
    :param mat: n-timepoints x m-cells matrix of traces
    :param n_trials: The number of trials
    :return: Trial average activity of shape (n-timepoints/n_trials) x m-cells
    """
    if mat.shape[0] % n_trials != 0:
        raise ValueError("Number of timepoints can't be divided into select number of trials")
    t_length = mat.shape[0] // n_trials
    return np.mean(mat.reshape((n_trials, t_length, mat.shape[1])), 0)


def rank_error(y_real: np.ndarray, prediction: np.ndarray):
    """
    Compute prediction rank error
    :param y_real: Matrix (samplesxfeatures) of real outputs
    :param prediction: Matrix (samplesxfeatures) of network predictions
    :return: Average rank error across all samples
    """
    nsamples = y_real.shape[0]
    if prediction.shape[0] != nsamples:
        raise ValueError("y_real and prediction need to have same number of samples")
    err_sum = 0
    for (y, p) in zip(y_real, prediction):
        r_y = np.unique(y, return_inverse=True)[1]
        r_p = np.unique(p, return_inverse=True)[1]
        err_sum += np.sum(np.abs(r_y - r_p))
    return err_sum / nsamples


def behavior_by_temperature(db_dict: dict, all_temps: np.ndarray, bins: np.ndarray):
    """
    Computes frequency of behavior selection by temperature
    :param db_dict: Debug dictionary created during simulation run
    :param all_temps: For each simulation position the occupied temperature
    :param bins: The bin-edges for dividing the temperature space
    :return: A dictionary with behaviors as keys and probability in each bin as values
    """
    selector = np.logical_and(db_dict["sel_behav"] != '', db_dict["sel_behav"] != 'N')
    behavior_types = np.unique(db_dict["sel_behav"][selector])
    occupancy_counts = np.histogram(all_temps, bins)[0].astype(np.float)
    result = {k: np.zeros(bins.size-1) for k in behavior_types}
    for behav in behavior_types:
        b_temps = db_dict["curr_temp"][db_dict["sel_behav"] == behav]
        result[behav] = np.histogram(b_temps, bins)[0].astype(np.float) / occupancy_counts
    return result


def behavior_by_delta_temp(db_dict: dict, bins: np.ndarray):
    """
    Computes frequency of behavior by delta-temperature achieved during the preceding bout
    :param db_dict: Debug dictionary created during simulation run
    :param bins: The bin-edges for dividing the bout delta temperature space
    :return: A dictionary with behaviors as keys and probability in each bin as values
    """
    selector = np.logical_and(db_dict["sel_behav"] != '', db_dict["sel_behav"] != 'N')
    behavior_types = np.unique(db_dict["sel_behav"][selector])
    all_behavs = db_dict["sel_behav"][selector]
    all_btemps = db_dict["curr_temp"][selector]
    all_deltas = np.zeros(all_btemps.size)
    for i in range(1, all_btemps.size):
        all_deltas[i] = all_btemps[i] - all_btemps[i-1]
    ad_counts = np.histogram(all_deltas, bins)[0].astype(np.float)
    result = {k: np.zeros(bins.size - 1) for k in behavior_types}
    for behav in behavior_types:
        b_dtemps = all_deltas[all_behavs == behav]
        result[behav] = np.histogram(b_dtemps, bins)[0].astype(np.float) / ad_counts
    return result


def _cluster_responses(response_mat, n_clusters, corr_cut=0.6):
    """
    Clusters the neuron responses using spectral clustering
    :param response_mat: The response matrix with all neuron responses
    :param n_clusters: The desired number of clusters
    :param corr_cut: The correlation cutoff to consider a given neuron to be part of a cluster
    :return:
        [0]: The cluster ids
        [1]: 3D embedding coordinates for plotting
    """
    # create trial average
    response_mat = trial_average(response_mat, 3)
    # compute pairwise correlations
    pw_corrs = np.corrcoef(response_mat.T)
    pw_corrs[np.isnan(pw_corrs)] = 0
    pw_corrs[pw_corrs < 0.2] = 0
    # perform spectral clustering
    spec_clust = SpectralClustering(n_clusters, affinity="precomputed")
    clust_ids = spec_clust.fit_predict(pw_corrs)
    spec_emb = SpectralEmbedding(3, affinity="precomputed")
    coords = spec_emb.fit_transform(pw_corrs)
    # use correlation to cluster centroids to determine final cluster membership
    regressors = np.zeros((response_mat.shape[0], n_clusters))
    for i in range(n_clusters):
        regressors[:, i] = np.mean(response_mat[:, clust_ids == i], 1)
    for i in range(response_mat.shape[1]):
        max_ix = -1
        max_corr = 0
        for j in range(n_clusters):
            c = np.corrcoef(response_mat[:, i], regressors[:, j])[0, 1]
            if c >= corr_cut and c > max_corr:
                max_ix = j
                max_corr = c
            clust_ids[i] = max_ix
    return clust_ids, coords


def cluster_activity(n_regs, all_cells, cluster_file=None):
    clust_ids, coords = None, None
    load_success = False
    if cluster_file is not None and os.path.exists(cluster_file):
        clfile = h5py.File(cluster_file, "r")
        # ensure that same amount of clusters were formed in the file
        if np.array(clfile["n_regs"]) == n_regs:
            clust_ids = np.array(clfile["clust_ids"])
            coords = np.array(clfile["coords"])
            # ensure that the same number of cells was clustered
            if clust_ids.size == coords.shape[0] and clust_ids.size == all_cells.shape[1]:
                load_success = True
        clfile.close()
    if not load_success:
        clust_ids, coords = _cluster_responses(all_cells, n_regs)
        if cluster_file is not None:
            clfile = h5py.File(cluster_file, "w")
            clfile.create_dataset("n_regs", data=n_regs)
            clfile.create_dataset("clust_ids", data=clust_ids)
            clfile.create_dataset("coords", data=coords)
            clfile.close()
    return clust_ids, coords
