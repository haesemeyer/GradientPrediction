#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Module of analysis specific implementations of persistent stores
"""

from core import PersistentStore, ModelData, GradientStandards
import pickle
import numpy as np
from hashlib import md5
from global_defs import GlobalDefs
from mo_types import MoTypes


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
    def __init__(self, db_file_name, std: GradientStandards, mo: MoTypes, read_only=False):
        """
        Creates a new simulation store
        :param db_file_name: The backend database filename
        :param std: Gradient standards for input normalization
        :param mo: Definition of model organism to use
        :param read_only: If true, no modifications will be made to the database
        """
        super().__init__(db_file_name, read_only)
        self.std = std
        self.mo = mo

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

    def _run_sim(self, model_path: str, sim_type: str, network_state: str, debug: bool, drop_list=None):
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
        mdata = ModelData(model_path)
        if network_state == "naive":
            chk = mdata.FirstCheckpoint
        else:
            chk = mdata.LastCheckpoint
        gpn = self.mo.network_model()
        gpn.load(mdata.ModelDefinition, chk)
        if sim_type == "r":
            sim = self.mo.rad_sim(gpn, self.std, **GlobalDefs.circle_sim_params)
        else:
            sim = self.mo.lin_sim(gpn, self.std, **GlobalDefs.lin_sim_params)
        sim.remove = drop_list
        if network_state == "bfevolve":
            ev_path = model_path + '/evolve/generation_weights.npy'
            weights = np.load(ev_path)
            w = np.mean(weights[-1, :, :], 0)
            sim.bf_weights = w
        if network_state == "ideal":
            return sim.run_ideal(GlobalDefs.n_steps)
        else:
            return sim.run_simulation(GlobalDefs.n_steps, debug)

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
        mdir = self.model_dir_name(model_path)
        get_list = [mdir, sim_type, network_state] if drop_list is None else [mdir, sim_type, network_state,
                                                                              md5(str(drop_list).encode()).hexdigest()]
        pos = self._get_data(*get_list, "pos")
        if pos is not None:
            return pos
        else:
            pos = self._run_sim(model_path, sim_type, network_state, False, drop_list)
            self._set_data(pos, *get_list, "pos")
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
        mdir = self.model_dir_name(model_path)
        get_list = [mdir, sim_type, network_state] if drop_list is None else [mdir, sim_type, network_state,
                                                                              md5(str(drop_list).encode()).hexdigest()]
        db_pickle = self._get_data(*get_list, "debug")
        pos = self._get_data(*get_list, "pos")
        if db_pickle is not None and pos is not None:
            deb_dict = pickle.loads(db_pickle)
            return pos, deb_dict
        else:
            pos, dbdict = self._run_sim(model_path, sim_type, network_state, True, drop_list=drop_list)
            self._set_data(pos, *get_list, "pos")
            self._set_data(np.void(pickle.dumps(dbdict)), *get_list, "debug")
            return pos, dbdict


class ActivityStore(ModelStore):
    """
    Hdf5 backed store of network cell activity data
    """
    def __init__(self, db_file_name, std: GradientStandards, mo: MoTypes, read_only=False):
        """
        Creates a new ActivityStore
        :param db_file_name: The backend database filename
        :param std: Gradient standards for input normalization
        :param mo: Definition of model organism to use
        :param read_only: If true, no modifications will be made to the database
        """
        super().__init__(db_file_name, read_only)
        self.std = std
        self.mo = mo

    def _compute_cell_responses(self, model_dir, temp, network_id, drop_list=None):
        """
        Loads a model and computes the temperature response of all neurons returning response matrix
        :param model_dir: The directory of the network model
        :param temp: The temperature input to test on the network
        :param network_id: Numerical id of the network to later relate units back to a network
        :param drop_list: Optional det_drop dictionary of lists of units to keep or drop
        :return:
            [0]: n-timepoints x m-neurons matrix of responses
            [1]: 3 x m-neurons matrix with network_id in row 0, layer index in row 1, and unit index in row 2
        """
        mdata = ModelData(model_dir)
        gpn_trained = self.mo.network_model()
        gpn_trained.load(mdata.ModelDefinition, mdata.LastCheckpoint)
        # prepend lead-in to stimulus
        lead_in = np.full(gpn_trained.input_dims[2] - 1, np.mean(temp[:10]))
        temp = np.r_[lead_in, temp]
        act_dict = gpn_trained.unit_stimulus_responses(temp, None, None, self.std, det_drop=drop_list)
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

    def get_cell_responses(self, model_path: str, temperature: np.ndarray, network_id: int, drop_list=None):
        """
        Obtain cell responses of all units in the given model for the given temperature stimulus
        :param model_path: The full path to the network model
        :param temperature: The temperature stimulus to present to the network
        :param network_id: An assigned numerical id of the network to later relate units back to a network
        :param drop_list: Optional det_drop dictionary of lists of units to keep or drop
        :return:
            [0]: n-timepoints x m-neurons matrix of responses
            [1]: 3 x m-neurons matrix with network_id in row 0, layer index in row 1 and unit index in row 2
        """
        stim_hash = md5(temperature).hexdigest()
        mdir = self.model_dir_name(model_path)
        getlist = [mdir, stim_hash] if drop_list is None else [mdir, stim_hash,
                                                               md5(str(drop_list).encode()).hexdigest()]
        activities = self._get_data(*getlist, "activities")
        id_mat = self._get_data(*getlist, "id_mat")
        if activities is not None and id_mat is not None:
            # re-assign network id
            id_mat[0, :] = network_id
            return activities, id_mat
        activities, id_mat = self._compute_cell_responses(model_path, temperature, network_id)
        self._set_data(activities, *getlist, "activities")
        self._set_data(id_mat, *getlist, "id_mat")
        return activities, id_mat
