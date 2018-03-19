#  Copyright 2017 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Commonly used helper functions
"""

import tensorflow as tf
import os
from warnings import warn
import numpy as np
import h5py
import matplotlib.pyplot as pl
import seaborn as sns
from global_defs import GlobalDefs


# Functions
def ca_convolve(trace, ca_timeconstant, frame_rate, kernel=None):
    """
    Convolves a trace with a decaying calcium kernel
    :param trace: The activity trace to convolve
    :param ca_timeconstant: The timeconstant of the calcium indicator
    :param frame_rate: The original frame-rate to relate samples to the time constant
    :param kernel: Optionally a pre-computed kernel in which case ca_timeconstant and frame_rate will be ignored
    :return: The convolved trace
    """

    def ca_kernel(tau, framerate):
        """
        Creates a calcium decay kernel for the given frameRate
        with the given half-life in seconds
        """
        fold_length = 5  # make kernel length equal to 5 half-times (decay to 3%)
        klen = int(fold_length * tau * framerate)
        tk = np.linspace(0, fold_length * tau, klen, endpoint=False)
        k = 2 ** (-1 * tk / tau)
        k = k / k.sum()
        return k

    if ca_timeconstant == 0 and kernel is None:
        return trace
    if kernel is None:
        kernel = ca_kernel(ca_timeconstant, frame_rate)
    return np.convolve(trace, kernel)[:trace.size]


def create_weight_var(name, shape, w_decay=None, loss_collection="losses", dale=False):
    """
    Creates a weight variable with optional weight decay initialized with sd = 1/size
    :param name: The name of the variable
    :param shape: The desired shape
    :param w_decay: None or L2 loss term if weight decay is desired
    :param loss_collection: The name of the collection to which loss should be added
    :param dale: If set to true enforce L1 loss on those weights going out of a unit whose sign is opposite
        of the average sign of weights from that unit
    :return: The weight variable
    """
    var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
    if dale:
        # compute a loss that penalizes wheights coming from one unit which have a sign
        # opposite of the average
        av_w = tf.sign(tf.reduce_mean(var, 1, keepdims=True))
        # positive value for weights whose sign is opposite of sign of the mean
        dale_loss = tf.reduce_sum(tf.maximum(-av_w * var, 0), name="dale_w_loss_"+name)
        tf.add_to_collection(loss_collection, 1e-4*dale_loss)
    if w_decay is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), w_decay, name="l2_w_loss_"+name)
        tf.add_to_collection(loss_collection, weight_decay)
    return var


def create_bias_var(name, shape):
    """
    Creates a bias variable initialized to 1.0 to avoid dead on init units when using ReLU
    :param name: The name of the variable
    :param shape: The desired shape
    :return: The bias variable
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def create_conv2d(name, x, W, mode="VALID"):
    """
    Create 2D convolution with stride 1
    :param name: The name of the operation output
    :param x: The input tensor
    :param W: The convolution weights of desired shape
    :param mode: The convolution mode 'VALID' or 'SAME'
    :return: The convolution operation
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=mode, name=name)


def create_meanpool2d(name, x, ax1, ax2):
    """
    Create average 2D pooling operation (i.e. binning operation leaving batch and channel axis untouched)
    :param name: The name of the operation output
    :param x: The input tensor
    :param ax1: The amount of pooling along the first 2D axis
    :param ax2: The amount of pooling along the second 2D axis
    :return: The pooling operation
    """
    return tf.nn.avg_pool(x, ksize=[1, ax1, ax2, 1], strides=[1, ax1, ax2, 1], padding='SAME', name=name)


def get_loss(labels, predictions, loss_collection="losses"):
    """
    Computes the total loss as the mean squared error loss of the current prediction and
    all the weight decay losses in the model
    :param labels: The real output values
    :param predictions: The output predictions
    :param loss_collection: The name of the collection containing all losses
    :return: The total loss tensor
    """
    # Calculate batch average mean squared loss
    sq_loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)
    joint_loss = tf.reduce_sum(sq_loss)
    tf.add_to_collection(loss_collection, joint_loss)
    return tf.add_n(tf.get_collection(loss_collection), name='total_loss'), joint_loss


def create_train_step(total_loss):
    """
        Creates a training step of the model given the labels and predictions tensor
        :param total_loss: The total loss of the model
        :return: The train step
    """
    return tf.train.AdamOptimizer(1e-4).minimize(total_loss)


def indexing_matrix(triggers: np.ndarray, past: int, future: int, input_length: int):
    """
    Builds an indexing matrix from an vector of triggers
    :param triggers: The elements on which to trigger (timepoint 0)
    :param past: The number of elements into the past to include
    :param future: The number of elements into the future to include
    :param input_length: The total length of the array to eventually index to determine valid triggers
    :return:
        [0]: The n_valid_triggers x (past+1+future) trigger matrix
        [1]: The number of triggers that have been cut out because they would have included indices < 0
        [2]: The number of triggers that have been cut out because they would have included indices >= input_length
    """
    if triggers.ndim > 1:
        raise ValueError("Triggers has to be 1D vector")
    to_take = np.r_[-past:future + 1][None, :]
    t = triggers[:, None]
    # construct trigger matrix
    index_mat = np.repeat(t, to_take.size, 1) + np.repeat(to_take, t.size, 0)
    # identify front and back rows that need to be removed
    cut_front = np.sum(np.sum(index_mat < 0, 1) > 0, 0)
    cut_back = np.sum(np.sum(index_mat >= input_length, 1) > 0, 0)
    # remove out-of-bounds rows
    if cut_back > 0:
        return index_mat[cut_front:-cut_back, :].astype(int), cut_front, cut_back
    else:
        return index_mat[cut_front:, :].astype(int), cut_front, 0


# Classes
class GradientStandards:
    """
    Lightweight wrapper of only standardizations used in a gradient data object
    """
    def __init__(self, temp_mean, temp_std, disp_mean, disp_std, ang_mean, ang_std):
        """
        Creates a new GradientStandards object
        :param temp_mean: The temperature average
        :param temp_std: The temperature standard deviation
        :param disp_mean: The displacement average
        :param disp_std: The displacement standard deviation
        :param ang_mean: The angle average
        :param ang_std: The angle standard
        """
        self.temp_mean = temp_mean
        self.temp_std = temp_std
        self.disp_mean = disp_mean
        self.disp_std = disp_std
        self.ang_mean = ang_mean
        self.ang_std = ang_std


class NotInitialized(Exception):
    def __init__(self, message):
        super().__init__(message)


class NetworkModel:
    """
    Base class for neuronal network models. Offers very basic shared functionality
    """
    def __init__(self, use_dale_constraint):
        """
        Creates a new NetworkModel
        :param use_dale_constraint: If set to true, 1/2 of network units can only provide inhibition 1/2 only excitation
        """
        self.initialized = False
        if use_dale_constraint:
            warn("Current implementation of dale constraint hinders network training")
        self.use_dale_constraint = use_dale_constraint
        # set training defaults
        self.w_decay = 1e-4
        self.keep_train = 0.5
        assert GlobalDefs.frame_rate % GlobalDefs.model_rate == 0
        self.t_bin = GlobalDefs.frame_rate // GlobalDefs.model_rate  # bin input down to 5Hz
        self.binned_size = GlobalDefs.frame_rate * GlobalDefs.hist_seconds // self.t_bin
        self._x_in = None  # network inputs
        # our branches
        self._branches = None
        # the number of our convolution layers
        self.n_conv_layers = None
        # deterministic removal units
        self._det_remove = {}
        # our graph object
        self._graph = None  # type: tf.Graph
        # our session object
        self._session = None  # type: tf.Session
        # saver object to save progress
        self._saver = None  # type: tf.train.Saver

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear()
        return False  # re-raise any previous exceptions

    # Protected API
    def _check_init(self):
        """
        Checks if network is initialized and raises exception otherwise
        """
        if not self.initialized:
            raise NotInitialized("Can't perform operation before performing setup of graph.")

    def _create_convolution_layer(self, branch, prev_out) -> tf.Tensor:
        """
        Creates a convolution layer
        :param branch: The branch on which to create the layer
        :param prev_out: The previous output tensor (= input to the convolution)
        :return: The flattened output of the convolution operation
        """
        if 't' in self._branches:
            len_0 = 1  # branched network, only one input to our convolution layer
        else:
            len_0 = 3  # fully mixed network, all three inputs are convolved together
        w_conv1 = create_weight_var(self.cvn("WEIGHT", branch, -1), [len_0, self.binned_size, 1, self.n_conv_layers])
        b_conv1 = create_bias_var(self.cvn("BIAS", branch, -1), [self.n_conv_layers])
        conv1 = create_conv2d(self.cvn("CONV", branch, -1), prev_out, w_conv1)
        cname = self.cvn("HIDDEN", branch, -1)
        h_conv1 = tf.nn.relu(conv1 + b_conv1, cname)
        h_conv1_flat = tf.reshape(h_conv1, [-1, self.n_conv_layers], cname+"_flat")
        return h_conv1_flat

    def _create_hidden_layer(self, branch: str, index: int, prev_out: tf.Tensor, n_units: int) -> tf.Tensor:
        """
        Creates a hidden layer in the network
        :param branch: The branch that layer belongs to
        :param index: The 0-based index of the hidden layer within the branch
        :param prev_out: The output tensor of the previous layer
        :param n_units: The number of units in this layer
        :return: The hidden layer activations
        """
        if branch not in self._branches:
            raise ValueError("branch {0} is not valid. Has to be one of {1}".format(branch, self._branches))
        w = create_weight_var(self.cvn("WEIGHT", branch, index), [prev_out.shape[1].value, n_units], self.w_decay,
                              dale=self.use_dale_constraint)
        b = create_bias_var(self.cvn("BIAS", branch, index), [n_units])
        dr = self._det_remove[branch][index]
        scale = n_units / tf.reduce_sum(dr)
        h = tf.nn.relu((tf.matmul(prev_out, w) + b) * dr * scale, self.cvn("HIDDEN", branch, index))
        return h

    # Public API
    def load(self, meta_file: str, checkpoint_file: str):
        """
        Loads model definitions from model description file and populates data from given checkpoint
        :param meta_file: The model definition file
        :param checkpoint_file: The saved model checkpoint (weights, etc.)
        """
        self.clear()
        self._graph = tf.Graph()

    def clear(self):
        """
        Clears the network graph
        """
        if not self.initialized:
            return
        # close session object if it exists
        if self._session is not None:
            self._session.close()
            self._session = None
        self._graph = None
        self._saver = None
        # mark network as not initialized
        self.initialized = False

    def init_variables(self):
        """
        Runs global variable initializer, resetting all values
        """
        self._check_init()
        with self._graph.as_default():
            self._session.run(tf.global_variables_initializer())

    def save_state(self, chkpoint_file, index, save_meta=True) -> str:
        """
        Saves the current state of the network to a checkpoint
        :param chkpoint_file: The base chkpoint file name
        :param index: The index of the current chkpoint
        :param save_meta: Whether to save the meta-graph as well or not
        :return: The full chkpoint filename and path
        """
        self._check_init()
        with self._graph.as_default():
            if self._saver is None:
                self._saver = tf.train.Saver(max_to_keep=None)  # never delete chkpoint files
            return self._saver.save(self._session, chkpoint_file, global_step=index, write_meta_graph=save_meta)

    @property
    def input_dims(self):
        """
        The network's input dimensions
        """
        self._check_init()
        with self._graph.as_default():
            return self._x_in.shape.as_list()

    @staticmethod
    def cvn(vartype: str, branch: str, index: int) -> str:
        """
        Creates a reproducible variable name for layer specific variables
        :param vartype: The variable type (WEIGHT, BIAS, HIDDEN, DROP, REMOVE, CONV)
        :param branch: The branch on which to create variable
        :param index: The layer index of the variable
        :return: The variable name
        """
        vartype = vartype.upper()
        vartypes = ["WEIGHT", "BIAS", "HIDDEN", "DROP", "REMOVE", "CONV", "OUTPUT"]
        if vartype not in vartypes:
            raise ValueError("Unknown vartype {0}. Has to be in {1}".format(vartype, vartypes))
        return "{0}_{1}_{2}".format(vartype, branch, index)


class GpNetworkModel(NetworkModel):
    """
    Base class of branched gradient prediction network models
    """
    def __init__(self, use_dale_constraint):
        """
        Creates a new GpNetworkModel
        """
        super().__init__(use_dale_constraint)
        # initialize fields that will be populated later
        self.n_units = None
        self.n_layers_branch = None
        self.n_layers_mixed = None
        self._n_mixed_dense = None
        self._n_branch_dense = None
        # model fields that are later needed for parameter feeding
        self._keep_prob = None  # drop-out keep probability
        self._y_ = None  # true responses (for training)
        # network output
        self._m_out = None  # type: tf.Tensor
        # the square loss (loss w.o. weight decay)
        self._sq_loss = None  # type: tf.Tensor
        # total loss across the network
        self._total_loss = None  # type: tf.Tensor
        # the training step to train the network
        self._train_step = None  # type: tf.Operation

    # Protected API
    def _create_unit_lists(self):
        """
        Creates lists of hidden unit counts and branch list according to network configuration
        """
        self._det_remove = {}
        self._n_mixed_dense = [self.n_units[1]] * self.n_layers_mixed
        if self.n_layers_branch == 0:
            self._branches = ['m', 'o']  # mixed network
        else:
            self._branches = ['t', 's', 'a', 'm', 'o']  # single input network
            self._n_branch_dense = [self.n_units[0]] * self.n_layers_branch

    def _create_real_out_placeholder(self) -> tf.Tensor:
        """
        Creates the 2D placeholder for the true labels
        Abstract method in this baseclass
        """
        raise NotImplementedError("ABSTRACT")

    def _create_output(self, prev_out: tf.Tensor) -> tf.Tensor:
        """
        Creates the output layer for reporting predicted temperature of all behaviors
        Abstract method in this baseclass
        """
        raise NotImplementedError("ABSTRACT")

    def _create_branch(self, branch: str, prev_out: tf.Tensor) -> tf.Tensor:
        """
        Creates a branch of the network
        :param branch: The name of the branch
        :param prev_out: The output of the previous layer
        :return: The output of the branch
        """
        if branch not in self._branches:
            raise ValueError("branch {0} is not valid. Has to be one of {1}".format(branch, self._branches))
        n_layers = self.n_layers_mixed if branch == 'm' else self.n_layers_branch
        last = prev_out
        for i in range(n_layers):
            last = self._create_hidden_layer(branch, i, last,
                                             self._n_mixed_dense[i] if branch == 'm' else self._n_branch_dense[i])
            last = tf.nn.dropout(last, self._keep_prob, name=self.cvn("DROP", branch, i))
        return last

    def _create_feed_dict(self, x_vals, y_vals=None, keep=1.0, removal=None) -> dict:
        """
        Create network feed dict
        :param x_vals: The network input values
        :param y_vals: True output values for training (optional)
        :param keep: The dropout probability for keeping all units
        :param removal: Deterministic keep/removal vectors
        :return: The feeding dict to pass to the network
        """
        f_dict = {self._x_in: x_vals, self._keep_prob: keep}
        if y_vals is not None:
            f_dict[self._y_] = y_vals
        # Fill deterministic removal part of feed dict
        for b in self._branches:
            for i, dr in enumerate(self._det_remove[b]):
                s = dr.shape[0].value
                if removal is None or b not in removal:
                    f_dict[dr] = np.ones(s, dtype=np.float32)
                else:
                    if removal[b][i].size != s:
                        raise ValueError("removal in branch {0} layer {1} does not have required size of {2}".format(b,
                                                                                                                     i,
                                                                                                                     s))
                    f_dict[dr] = removal[b][i]
        return f_dict

    # Public API
    def setup(self, n_conv_layers: int, n_units, n_layers_branch: int, n_layers_mixed: int):
        """
        Creates the network graph from scratch according to the given specifications
        :param n_conv_layers: The number of convolutional layers per input branch
        :param n_units: The number of units in each hidden layer or 2 element list for units in branch and mix
        :param n_layers_branch: The number of hidden layers in each branch (can be 0 for full mixing)
        :param n_layers_mixed: The number of hidden layers in the mixed part of the model
        """
        self.clear()
        # ingest parameters
        if n_layers_mixed < 1:
            raise ValueError("Network needs at least on mixed hidden layer")
        if n_layers_branch < 0:
            raise ValueError("Number of branch layers can't be negative")
        self.n_conv_layers = n_conv_layers
        if type(n_units) is not list:
            self.n_units = [n_units] * 2
        else:
            if len(n_units) != 2:
                raise ValueError("n_units should either be scalar or a 2-element list")
            self.n_units = n_units
        self.n_layers_branch = n_layers_branch
        self.n_layers_mixed = n_layers_mixed
        self._create_unit_lists()
        self._graph = tf.Graph()
        with self._graph.as_default():
            # create deterministic removal units
            for b in self._branches:
                if b == 'm':
                    self._det_remove[b] = [tf.placeholder(tf.float32, shape=[self._n_mixed_dense[i]],
                                                          name=self.cvn("REMOVE", b, i))
                                           for i in range(self.n_layers_mixed)]
                else:
                    self._det_remove[b] = [tf.placeholder(tf.float32, shape=[self._n_branch_dense[i]],
                                                          name=self.cvn("REMOVE", b, i))
                                           for i in range(self.n_layers_branch)]
            # dropout probability placeholder
            self._keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            # model input: BATCHSIZE x (Temp,Move,Turn) x HISTORYSIZE x 1 CHANNEL
            self._x_in = tf.placeholder(tf.float32, [None, 3, GlobalDefs.frame_rate*GlobalDefs.hist_seconds, 1], "x_in")
            # real outputs: BATCHSIZE x (dT(Stay), dT(Straight), dT(Left), dT(Right))
            self._y_ = self._create_real_out_placeholder()
            # data binning layer
            xin_pool = create_meanpool2d("xin_pool", self._x_in, 1, self.t_bin)
            # the input convolution depends on the network structure: branched or fully mixed
            if 't' in self._branches:
                # branched network - split input into temperature, speed and angle
                x_1, x_2, x_3 = tf.split(xin_pool, num_or_size_splits=3, axis=1, name="input_split")
                # create convolution and deep layer for each branch
                time = self._create_convolution_layer('t', x_1)
                time = self._create_branch('t', time)
                speed = self._create_convolution_layer('s', x_2)
                speed = self._create_branch('s', speed)
                angle = self._create_convolution_layer('a', x_3)
                angle = self._create_branch('a', angle)
                # combine branch outputs and create mix branch
                mix = tf.concat([time, speed, angle], 1, self.cvn("HIDDEN", 'm', -1))
                mix = self._create_branch('m', mix)
            else:
                # fully mixed network
                mix = self._create_convolution_layer('m', xin_pool)
                mix = self._create_branch('m', mix)
            self._m_out = self._create_output(mix)
            # create and store losses and training step
            self._total_loss, self._sq_loss = get_loss(self._y_, self._m_out)
            self._train_step = create_train_step(self._total_loss)
            # store our training operation
            tf.add_to_collection('train_op', self._train_step)
            # create session
            self._session = tf.Session()
            # mark network as initialized
            self.initialized = True
            # intialize all variables
            self.init_variables()

    def load(self, meta_file: str, checkpoint_file: str):
        """
        Loads model definitions from model description file and populates data from given checkpoint
        :param meta_file: The model definition file
        :param checkpoint_file: The saved model checkpoint (weights, etc.)
        """
        super().load(meta_file, checkpoint_file)
        with self._graph.as_default():
            # restore graph and variables
            self._session = tf.Session()
            saver = tf.train.import_meta_graph(meta_file)
            saver.restore(self._session, checkpoint_file)
            graph = self._session.graph
            self._m_out = graph.get_tensor_by_name(self.cvn("OUTPUT", 'o', 0)+":0")
            self._x_in = graph.get_tensor_by_name("x_in:0")
            self._keep_prob = graph.get_tensor_by_name("keep_prob:0")
            self._y_ = graph.get_tensor_by_name("y_:0")
            # collect deterministic removal units and use these
            # to determine which branches exist and how many layers they have
            possible_branches = ['t', 's', 'a', 'm', 'o']
            self._branches = []
            self._det_remove = {}
            for b in possible_branches:
                try:
                    graph.get_tensor_by_name(self.cvn("REMOVE", b, 0)+":0")
                    # we found layer 0 in this branch so it exists
                    self._branches.append(b)
                    self._det_remove[b] = []
                    i = 0
                    try:
                        while True:
                            self._det_remove[b].append(graph.get_tensor_by_name(self.cvn("REMOVE", b, i)+":0"))
                            i += 1
                    except KeyError:
                        pass
                except KeyError:
                    continue
            if 't' in self._branches:
                self.n_layers_branch = len(self._det_remove['t'])
            else:
                self.n_layers_branch = 0
            self.n_units = [0, 0]
            self.n_layers_mixed = len(self._det_remove['m'])
            self._n_mixed_dense = [self._det_remove['m'][i].shape[0].value for i in range(self.n_layers_mixed)]
            self.n_units[1] = self._n_mixed_dense[0]
            if self.n_layers_branch > 0:
                self._n_branch_dense = [self._det_remove['t'][i].shape[0].value for i in range(self.n_layers_branch)]
                self.n_units[0] = self._n_branch_dense[0]
            else:
                self._n_branch_dense = []
            # retrieve training step
            self._train_step = graph.get_collection("train_op")[0]
            # set up squared loss calculation
            self._sq_loss = tf.losses.mean_squared_error(labels=self._y_, predictions=self._m_out)
        self.initialized = True
        # use convolution data biases to get number of convolution layers
        conv_biases = self.convolution_data[1]
        self.n_conv_layers = conv_biases.popitem()[1].shape[0]

    def clear(self):
        """
        Clears the network graph
        """
        if not self.initialized:
            return
        super().clear()
        self.n_conv_layers = None
        self.n_units = None
        self.n_layers_branch = None
        self.n_layers_mixed = None
        # mark network as not initialized
        self.initialized = False

    def train(self, xbatch, ybatch, keep=0.5):
        """
        Runs a training step on the given batches
        :param xbatch: The input of the training batch
        :param ybatch: The true labels of the training batch
        :param keep: The keep probability of each unit
        """
        self._check_init()
        with self._graph.as_default():
            self._train_step.run(self._create_feed_dict(xbatch, ybatch, keep), self._session)

    def get_squared_loss(self, xbatch, ybatch, keep=1) -> float:
        """
        Computes the square loss over the given batch
        :param xbatch: The batch input
        :param ybatch: The true labels of the batch
        :param keep: The keep probability of each unit
        :return: The square loss
        """
        self._check_init()
        with self._graph.as_default():
            return self._sq_loss.eval(self._create_feed_dict(xbatch, ybatch, keep), self._session)

    def predict(self, xbatch, keep=1.0, det_drop=None) -> np.ndarray:
        """
        Uses the network to predict output given the input
        :param xbatch: The network input
        :param keep: The keep probability of each unit
        :param det_drop: The deterministic keep/drop of each unit
        :return: The network output
        """
        self._check_init()
        with self._graph.as_default():
            return self._m_out.eval(self._create_feed_dict(xbatch, keep=keep, removal=det_drop), session=self._session)

    def test_error_distributions(self, test_data):
        """
        For the network returns the distribution of squared losses and rank errors across test data
        :param test_data: The test data to evaluate the model
            [0]: For each datapoint in test_data the squared error loss
            [1]: For each datapoint in test_data the rank error
        """
        self._check_init()
        with self._graph.as_default():
            sq_errors = np.full(test_data.data_size, -1)
            rank_errors = np.full(test_data.data_size, -1)
            for i in range(test_data.data_size):
                xbatch, ybatch = test_data.training_batch(1)
                pred = self.predict(xbatch, 1.0)
                sq_errors[i] = (np.sum((ybatch - pred) ** 2))
                rank_real = np.unique(ybatch, return_inverse=True)[1]
                rank_pred = np.unique(pred, return_inverse=True)[1]
                rank_errors[i] = np.sum(np.abs(rank_real - rank_pred))
        return sq_errors, rank_errors

    def unit_stimulus_responses(self, temperature, speed, angle, standardizations: GradientStandards) -> dict:
        """
        Computes and returns the responses of each unit in the network in response to a stimulus
        :param temperature: The temperature stimulus in C (can be None for clamping to 0)
        :param speed: The speed input in pixels per timestep (can be None for clamping to 0)
        :param angle: The angle input in degrees per timestep (can be None for clamping to 0)
        :param standardizations: Object that provides mean and standard deviation for each input
        :return: Branch-wise dictionary of lists with n_hidden elements, each an array of time x n_units activations
        """
        self._check_init()
        with self._graph.as_default():
            # ensure that at least one stimulus was provided that all have same size and standardize them
            if any((temperature is not None, speed is not None, angle is not None)):
                sizes = [x.size for x in (temperature, speed, angle) if x is not None]
                if any([s != sizes[0] for s in sizes]):
                    raise ValueError("All given inputs must have same length")
                if temperature is None:
                    temperature = np.zeros(sizes[0], np.float32)
                else:
                    temperature = (temperature-standardizations.temp_mean) / standardizations.temp_std
                if speed is None:
                    speed = np.zeros(sizes[0], np.float32)
                else:
                    speed = (speed - standardizations.disp_mean) / standardizations.disp_std
                if angle is None:
                    angle = np.zeros(sizes[0], np.float32)
                else:
                    angle = (angle - standardizations.ang_mean) / standardizations.ang_std
            else:
                raise ValueError("At least one input needs to be given")
            history = self.input_dims[2]
            activity = {}

            ix = indexing_matrix(np.arange(temperature.size), history - 1, 0, temperature.size)[0]
            model_in = np.zeros((ix.shape[0], 3, history, 1))
            model_in[:, 0, :, 0] = temperature[ix]
            model_in[:, 1, :, 0] = speed[ix]
            model_in[:, 2, :, 0] = angle[ix]
            for b in self._branches:
                if b == 'o':
                    activity[b] = [self.predict(model_in)]
                    continue
                n_layers = self.n_layers_mixed if b == 'm' else self.n_layers_branch
                for i in range(n_layers):
                    h = self._session.graph.get_tensor_by_name(self.cvn("HIDDEN", b, i)+":0")
                    fd = self._create_feed_dict(model_in, keep=1.0)
                    if b in activity:
                        activity[b].append(h.eval(feed_dict=fd, session=self._session))
                    else:
                        activity[b] = [h.eval(feed_dict=fd, session=self._session)]
            return activity

    def branch_output(self, branch_name, xbatch, det_drop=None) -> np.ndarray:
        """
        Computes the activations of all units in the last hidden layer of the given branch
        :param branch_name: The name of the branch ('t', 's', 'a', 'm')
        :param xbatch: The network input
        :param det_drop: The deterministic keep/drop of each unit
        :return: The activations of the last layer of reach branch for the given inputs
        """
        self._check_init()
        if branch_name not in self._branches:
            raise ValueError("Branch '{0}' is not present in this network. Has to be one of {1}".format(branch_name,
                                                                                                        self._branches))
        # obtain the name of the last hidden layer of the given branch
        tensor_name = self.cvn("HIDDEN", branch_name,
                               self.n_layers_branch-1 if branch_name != 'm' else self.n_layers_mixed-1)
        tensor_name = tensor_name + ":0"
        with self._graph.as_default():
            fd = self._create_feed_dict(xbatch, removal=det_drop)
            tensor = self._graph.get_tensor_by_name(tensor_name)
            layer_out = tensor.eval(fd, session=self._session)
        return layer_out

    @property
    def convolution_data(self):
        """
        The weights and biases of the convolution layer(s)
        """
        self._check_init()
        with self._graph.as_default():
            if 't' in self._branches:
                to_get = ['t', 's', 'a']
            else:
                to_get = ['m']
            g = self._session.graph
            w = {tg: g.get_tensor_by_name(self.cvn("WEIGHT", tg, -1)+":0").eval(session=self._session) for tg in to_get}
            b = {tg: g.get_tensor_by_name(self.cvn("BIAS", tg, -1) + ":0").eval(session=self._session) for tg in to_get}
        return w, b


class ZfGpNetworkModel(GpNetworkModel):
    """
    Class representing zebrafish gradient prediction network model
    """
    def __init__(self, use_dale_constraint=False):
        """
        Creates a new ZfGpNetworkModel
        """
        super().__init__(use_dale_constraint)

    # Private API
    def _create_output(self, prev_out: tf.Tensor) -> tf.Tensor:
        """
        Creates the output layer for reporting predicted temperature of all four behaviors
        :param prev_out: The output of the previous layer
        :return: output
        """
        w = create_weight_var(self.cvn("WEIGHT", 'o', 0), [prev_out.shape[1].value, 4], self.w_decay,
                              dale=self.use_dale_constraint)
        b = create_bias_var(self.cvn("BIAS", 'o', 0), [4])
        out = tf.add(tf.matmul(prev_out, w), b, name=self.cvn("OUTPUT", 'o', 0))
        return out

    def _create_real_out_placeholder(self):
        """
        Creates placeholder variable for labels
        """
        # real outputs: BATCHSIZE x (dT(Stay), dT(Straight), dT(Left), dT(Right))
        return tf.placeholder(tf.float32, [None, 4], "y_")

    # Public API
    @staticmethod
    def plot_network(activations: dict, index: int):
        """
        Plots network structure with node darkness corresponding to its activation
        :param activations: Dictionary of branches with layer activation lists
        :param index: The frame in activations at which the network should be visualized
        :return: figure and axes object
        """
        def circle_pos(rownum, colnum, n_rows, n_cols):
            """
            Compute the relative position of one circle within a layer
            :param rownum: The row of the circle
            :param colnum: The column of the circle
            :param n_rows: The total number of rows in the layer
            :param n_cols: The total number of columns in the layer
            :return: The x,y position of the center
            """
            if rownum >= n_rows or colnum >= n_cols:
                raise ValueError("Row and column numbers can't be larger than totals")
            y_spread = (n_rows-1) * c_c_dist
            y_pos = - y_spread / 2 + rownum * c_c_dist
            x_spread = (n_cols-1) * c_c_dist
            x_pos = - x_spread / 2 + colnum * c_c_dist
            return x_pos, y_pos

        def layer_dim(values: np.ndarray):
            """
            Computes the width and height of the layer bounding box
            """
            l_size = values.size
            n_rows = (l_size - 1) // max_width + 1
            n_cols = max_width if l_size >= max_width else l_size
            boundx = 0 - (n_cols / 2) * (circle_dist + 2 * circle_rad) - circle_rad / 2
            boundy = 0 - (n_rows / 2) * (circle_dist + 2 * circle_rad) - circle_rad / 2
            boundw = (0 - boundx) * 2
            boundh = (0 - boundy) * 2
            return boundw, boundh

        def draw_layer(x_center, y_center, values: np.ndarray):
            """
            Creates artists for one whole layer of the network
            :param x_center: The x center coordinate of the layer
            :param y_center: They y center coordinate of the layer
            :param values: For each unit in the layer its normalized activation
            :return:
                [0] List of artists that draw this layer
                [1] (xmin, xmax, ymin, ymax) tuple of rectangle containing this layer
            """
            if np.any(values > 1) or np.any(values < 0):
                raise ValueError("values can't be smaller 0 or larger 1")
            arts = []
            l_size = values.size
            n_rows = (l_size-1) // max_width + 1
            n_cols = max_width if l_size >= max_width else l_size
            # compute bounding rectangle
            boundx = x_center - (n_cols / 2) * (circle_dist + 2*circle_rad) - circle_rad/2
            boundy = y_center - (n_rows / 2) * (circle_dist + 2*circle_rad) - circle_rad/2
            boundw = (x_center - boundx) * 2
            boundh = (y_center - boundy) * 2
            # draw units according to their activations
            for i, v in enumerate(values):
                x, y = circle_pos(i // max_width, i % max_width, n_rows, n_cols)
                x += x_center
                y += y_center
                arts.append(pl.Circle((x, y), circle_rad, color=(1-v, 1-v, 1-v)))
            return arts, (boundx, boundx+boundw, boundy, boundy+boundh)

        # compute normalization across whole timeseries
        max_width = 32  # maximum number of units in a row
        circle_rad = 10  # radius of each given circle
        circle_dist = 7  # the edge-to-edge distance of circles
        c_c_dist = circle_dist + 2*circle_rad  # the center-to-center distance btw. neighboring circles
        xcents = {'o': 0, 'm': 0, 't': -0.85*max_width*(c_c_dist+circle_rad), 's': 0,
                  'a': 0.85*max_width*(c_c_dist+circle_rad)}
        # the branch order from bottom to top
        order = ['o', 'm', 't', 's', 'a']
        # for each branch compute the y-center of its first layer
        thickness = {}
        for b in order:
            if b not in activations:
                thickness[b] = 0
            else:
                thickness[b] = 0
                for l in activations[b]:
                    thickness[b] += (layer_dim(l[index, :])[1] + c_c_dist * 2)
        ystarts = {}
        for b in order:
            if b == 'o':
                ystarts[b] = 0
            elif b == 'm':
                ystarts[b] = thickness[b] / 4
            else:
                ystarts[b] = ystarts['m'] + thickness['m'] + thickness[b] / 4
        all_artists = []
        fig_bounds = np.zeros(4)
        for b in order:
            if b not in activations:
                continue
            xc = xcents[b]
            prev_offset = 0
            for i, l in enumerate(activations[b]):
                yc = ystarts[b] + prev_offset
                prev_offset += layer_dim(l[index, :])[1] + c_c_dist
                minval = np.min(l, 0)
                diff = np.max(l, 0) - minval
                diff[diff == 0] = 0.1
                layer_arts, bounds = draw_layer(xc, yc, (l[index, :]-minval) / diff)
                all_artists += layer_arts
                # update figure bounds
                if fig_bounds[0] > bounds[0]:
                    fig_bounds[0] = bounds[0]
                if fig_bounds[1] < bounds[1]:
                    fig_bounds[1] = bounds[1]
                if fig_bounds[2] > bounds[2]:
                    fig_bounds[2] = bounds[2]
                if fig_bounds[3] < bounds[3]:
                    fig_bounds[3] = bounds[3]
        # create actual figure
        fig, ax = pl.subplots()
        for a in all_artists:
            ax.add_artist(a)
        ax.axis('square')
        # update limits
        ax.set_xlim(fig_bounds[0], fig_bounds[1])
        ax.set_ylim(fig_bounds[2], fig_bounds[3])
        sns.despine(fig, ax, True, True, True, True)
        ax.tick_params(
            axis='both',
            which='both',
            bottom='off',
            top='off',
            right='off',
            left='off',
            labelbottom='off',
            labelleft='off')
        return fig, ax


class CeGpNetworkModel(GpNetworkModel):
    """
    Class representing C elegans gradient prediction network model
    """
    def __init__(self, use_dale_constraint=False):
        """
        Creates a new CeGpNetworkModel
        """
        super().__init__(use_dale_constraint)

    # Private API
    def _create_real_out_placeholder(self):
        """
        Creates placeholder variable for labels
        """
        # real outputs: BATCHSIZE x (dT(Continue), dT(StrongTurn), dt(Pirouette), dT(LeftTurn), dT(RightTurn))
        return tf.placeholder(tf.float32, [None, 5], "y_")

    def _create_output(self, prev_out: tf.Tensor):
        """
        Creates the output layer for reporting predicted temperature of all five behaviors
        :param prev_out: The output of the previous layer
        :return: output
        """
        w = create_weight_var(self.cvn("WEIGHT", 'o', 0), [prev_out.shape[1].value, 5], self.w_decay,
                              dale=self.use_dale_constraint)
        b = create_bias_var(self.cvn("BIAS", 'o', 0), [5])
        out = tf.add(tf.matmul(prev_out, w), b, name=self.cvn("OUTPUT", 'o', 0))
        return out


class SimpleRLNetwork(NetworkModel):
    """
    Simple reinforcement learning network with only temperature input
    """
    def __init__(self, use_dale_constraint=False):
        """
        Create a new simple reinforcment learning network model
        """
        super().__init__(use_dale_constraint)
        # initialize fields that will be populated later
        self.n_units = None
        self.n_layers = None
        self._n_dense = None
        # model fields that are later needed for parameter feeding
        self._keep_prob = None  # drop-out keep probability
        self._reward = None  # reward delivered (for training)
        self._pick = None  # the behaviors that was chosen and led to the reward above
        self._responsible_out = None  # the output responsible for the current reward
        # network output
        self._value_out = None  # type: tf.Tensor
        self._log_value_out = None  # type: tf.Tensor
        # the reward based loss (loss w.o. weight decay)
        self._loss = None  # type: tf.Tensor
        # total loss across the network
        self._total_loss = None  # type: tf.Tensor
        # the training step to train the network
        self._train_step = None  # type: tf.Operation
        # create cash of uniform random numbers
        self._uni_cash = RandCash(1000, lambda s: np.random.rand(s))

    # Private API
    def _create_unit_lists(self):
        """
        Creates lists of hidden unit counts and branch list according to network configuration
        """
        self._det_remove = {}
        self._n_dense = [self.n_units] * self.n_layers
        self._branches = ['t', 'o']

    def _create_branch(self, branch: str, prev_out: tf.Tensor) -> tf.Tensor:
        """
        Creates a branch of the network
        :param branch: The name of the branch
        :param prev_out: The output of the previous layer
        :return: The output of the branch
        """
        if branch not in self._branches:
            raise ValueError("branch {0} is not valid. Has to be one of {1}".format(branch, self._branches))
        last = prev_out
        for i in range(self.n_layers):
            last = self._create_hidden_layer(branch, i, last, self._n_dense[i])
            last = tf.nn.dropout(last, self._keep_prob, name=self.cvn("DROP", branch, i))
        return last

    def _create_values(self, prev_out: tf.Tensor):
        """
        Creates the output layer for reporting predicted temperature of all four behaviors
        :param prev_out: The output of the previous layer
        :return: output, log_output
        """
        w = create_weight_var(self.cvn("WEIGHT", 'o', 0), [prev_out.shape[1].value, 2], self.w_decay)
        b = create_bias_var(self.cvn("BIAS", 'o', 0), [2])
        out = tf.nn.softmax((tf.matmul(prev_out, w) + b), name=self.cvn("OUTPUT", 'o', 0))
        log_out = tf.nn.log_softmax((tf.matmul(prev_out, w) + b), name=self.cvn("OUTPUT", 'o', -1))
        return out, log_out

    def _create_feed_dict(self, x_in, rewards=None, picks=None, keep=1.0, removal=None) -> dict:
        """
        Create network feed dict
        :param x_in: The network input value
        :param rewards: The delivered rewards
        :param picks: The chosen units (optional but needs to be present if reward != None)
        :param keep: The dropout probability for keeping all units
        :param removal: Deterministic keep/removal vectors
        :return: The feeding dict to pass to the network
        """
        f_dict = {self._x_in: x_in, self._keep_prob: keep}
        if rewards is not None:
            # augment rewards to 2D if necessary
            if rewards.ndim == 1:
                rewards = rewards[:, None]
            f_dict[self._reward] = rewards
            if picks is None or picks.size != rewards.size:
                raise ValueError("If rewards are provided, picks need to be provided with the same number of samples!")
            picks = np.c_[np.arange(picks.size)[:, None], picks[:, None]]
            f_dict[self._pick] = picks
        # Fill deterministic removal part of feed dict
        for b in self._branches:
            if b == 'o':
                continue
            for i, dr in enumerate(self._det_remove[b]):
                s = dr.shape[0].value
                if removal is None or b not in removal:
                    f_dict[dr] = np.ones(s, dtype=np.float32)
                else:
                    if removal[b][i].size != s:
                        raise ValueError("removal in branch {0} layer {1} does not have required size of {2}".format(b,
                                                                                                                     i,
                                                                                                                     s))
                    f_dict[dr] = removal[b][i]
        return f_dict

    # Public API
    def setup(self, n_conv_layers: int, n_units, n_layers: int):
        """
        Creates the network graph
        :param n_conv_layers: The number of convolutional layers operating on the input
        :param n_units: The number of units in each dense layer
        :param n_layers: The number of dense layers
        """
        self.clear()
        # ingest parameters
        if n_layers < 1:
            raise ValueError("Network needs at least one hidden layer")
        self.n_conv_layers = n_conv_layers
        if n_units < 1:
            raise ValueError("Each hidden layer needs at least one unit")
        self.n_units = n_units
        self.n_layers = n_layers
        self._create_unit_lists()
        self._graph = tf.Graph()
        with self._graph.as_default():
            # create deterministic removal units
            for b in self._branches:
                if b == 't':
                    self._det_remove[b] = [tf.placeholder(tf.float32, shape=[self._n_dense[i]],
                                                          name=self.cvn("REMOVE", b, i))
                                           for i in range(self.n_layers)]
            # dropout probability placeholder
            self._keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            # model input: NSAMPLES x Temp x HISTORYSIZE x 1 CHANNEL
            self._x_in = tf.placeholder(tf.float32, [None, 1, GlobalDefs.frame_rate*GlobalDefs.hist_seconds, 1], "x_in")
            # reward values: NSAMPLES x 1 (Note: Only picked behavior will get rewarded!)
            self._reward = tf.placeholder(tf.float32, [None, 1], name="reward")
            # sample index and index of the picked behavior: NSAMPLES x 2 (either 0 or 1, i.e. straight or turn)
            self._pick = tf.placeholder(tf.int32, [None, 2], name="pick")
            # data binning layer
            xin_pool = create_meanpool2d("xin_pool", self._x_in, 1, self.t_bin)
            # create convolution layer and deep layers
            conv = self._create_convolution_layer('t', xin_pool)
            deep_out = self._create_branch('t', conv)
            self._value_out, self._log_value_out = self._create_values(deep_out)
            self._responsible_out = tf.gather_nd(self._log_value_out, self._pick, "responsible_out")
            self._loss = -tf.reduce_sum(self._responsible_out * self._reward)
            tf.add_to_collection("losses", self._loss)
            # compute the total loss which includes our weight-decay
            self._total_loss = tf.add_n(tf.get_collection("losses"), name="total_loss")
            # create training step
            optimizer = tf.train.AdamOptimizer(1e-5)
            gradients, variables = zip(*optimizer.compute_gradients(self._total_loss))
            self._train_step = optimizer.apply_gradients(zip(gradients, variables))
            # self._train_step = tf.train.AdamOptimizer(1e-6).minimize(self._total_loss)
            # self._train_step = create_train_step(self._total_loss)
            # store our training operation
            tf.add_to_collection('train_op', self._train_step)
            # create session
            self._session = tf.Session()
            # mark network as initialized
            self.initialized = True
            # intialize all variables
            self.init_variables()

    def load(self, meta_file: str, checkpoint_file: str):
        """
        Loads model definitions from model description file and populates data from given checkpoint
        :param meta_file: The model definition file
        :param checkpoint_file: The saved model checkpoint (weights, etc.)
        """
        super().load(meta_file, checkpoint_file)
        with self._graph.as_default():
            # restore graph and variables
            self._session = tf.Session()
            saver = tf.train.import_meta_graph(meta_file)
            saver.restore(self._session, checkpoint_file)
            graph = self._session.graph
            self._value_out = graph.get_tensor_by_name(self.cvn("OUTPUT", 'o', 0)+":0")
            self._log_value_out = graph.get_tensor_by_name(self.cvn("OUTPUT", 'o', -1) + ":0")
            self._responsible_out = graph.get_tensor_by_name("responsible_out:0")
            self._x_in = graph.get_tensor_by_name("x_in:0")
            self._keep_prob = graph.get_tensor_by_name("keep_prob:0")
            self._reward = graph.get_tensor_by_name("reward:0")
            self._pick = graph.get_tensor_by_name("pick:0")
            self._branches = ['t', 'o']
            # collect deterministic removal units
            for b in self._branches:
                try:
                    graph.get_tensor_by_name(self.cvn("REMOVE", b, 0)+":0")
                    self._det_remove[b] = []
                    i = 0
                    try:
                        while True:
                            self._det_remove[b].append(graph.get_tensor_by_name(self.cvn("REMOVE", b, i)+":0"))
                            i += 1
                    except KeyError:
                        pass
                except KeyError:
                    continue
            self.n_layers = len(self._det_remove['t'])
            self._n_dense = [self._det_remove['t'][i].shape[0].value for i in range(self.n_layers)]
            self.n_units = self._n_dense[0]
            # retrieve training step
            self._train_step = graph.get_collection("train_op")[0]
            # set up loss calculation
            self._loss = -(tf.log(self._responsible_out) * self._reward)[0]
        self.initialized = True
        # use convolution data biases to get number of convolution layers
        conv_biases = self.convolution_data[1]
        self.n_conv_layers = conv_biases.popitem()[1].shape[0]

    def clear(self):
        """
        Clears the network graph
        """
        if not self.initialized:
            return
        super().clear()
        self.n_conv_layers = None
        self.n_units = None
        self.n_layers = None
        # mark network as not initialized
        self.initialized = False

    def train(self, x_in, reward: np.ndarray, pick: np.ndarray, keep=0.5):
        """
        Runs a single move training step
        :param x_in: The network input
        :param reward: The delivered reward
        :param pick: The chosen action
        :param keep: The keep probability of each unit
        """
        self._check_init()
        if reward.size > 1:
            warn("Providing more than one concurrent training sample is discouraged since credit assignment unclear.")
        with self._graph.as_default():
            fd = self._create_feed_dict(x_in, reward, pick, keep)
            self._train_step.run(fd, self._session)

    def get_values(self, x_in, keep=1.0, det_drop=None) -> np.ndarray:
        """
        Compute the predicted value of straight swim or turn
        :param x_in: The temperature history as network input
        :param keep: The keep probability of each unit
        :param det_drop: The deterministic keep/drop of each unit
        :return: 2-element vector of values for straight and turn
        """
        self._check_init()
        with self._graph.as_default():
            v = self._value_out.eval(self._create_feed_dict(x_in, keep=keep, removal=det_drop), session=self._session)
        return v

    def choose_action(self, x_in, p_explore=0.01, keep=1.0, det_drop=None):
        """
        Our policy. Calculate value of each action given input. Choose random action with p_explore probability
        and higher-valued action otherwise
        :param x_in: The temperature history as network input
        :param p_explore: Probability of choosing random action instead of following policy
        :param keep: The keep probability of each unit
        :param det_drop: The deterministic keep/drop of each unit
        :return: The index of the chosen action (straight=0, turn=1)
        """
        if self._uni_cash.next_rand() < p_explore:
            return np.random.randint(2)
        v = self.get_values(x_in, keep, det_drop).ravel()
        if np.any(np.isnan(v)):
            raise ValueError("Invalid probabilities returned")
        # we only have two actions - so these probabilities are in fact redundant at the moment
        # hence only compare to first
        dec = self._uni_cash.next_rand()
        if dec < v[0]:
            return 0
        else:
            return 1

    def final_hidden_output(self, xbatch, det_drop=None) -> np.ndarray:
        """
        Computes the activations of all units in the last hidden layer of the network
        :param xbatch: The network input
        :param det_drop: The deterministic keep/drop of each unit
        :return: The activations of the last hidden network layer
        """
        self._check_init()
        # obtain the name of the last hidden layer
        tensor_name = self.cvn("HIDDEN", 't', self.n_layers-1)
        tensor_name = tensor_name + ":0"
        with self._graph.as_default():
            fd = self._create_feed_dict(xbatch, removal=det_drop)
            tensor = self._graph.get_tensor_by_name(tensor_name)
            layer_out = tensor.eval(fd, session=self._session)
        return layer_out

    def unit_stimulus_responses(self, temperature, temp_mean, temp_std) -> dict:
        """
        Computes and returns the responses of each unit in the network in response to a stimulus
        :param temperature: The temperature stimulus in C
        :param temp_mean: The temperature average when the network was trained
        :param temp_std: The temperature standard deviation when the network was trained
        :return: Branch-wise dictionary of lists with n_hidden elements, each an array of time x n_units activations
        """
        self._check_init()
        temperature = (temperature - temp_mean) / temp_std
        with self._graph.as_default():
            history = self.input_dims[2]
            activity = {}
            ix = indexing_matrix(np.arange(temperature.size), history - 1, 0, temperature.size)[0]
            model_in = np.zeros((ix.shape[0], 1, history, 1))
            model_in[:, 0, :, 0] = temperature[ix]
            for b in self._branches:
                if b == 'o':
                    activity[b] = [self.get_values(model_in)]
                    continue
                for i in range(self.n_layers):
                    h = self._session.graph.get_tensor_by_name(self.cvn("HIDDEN", b, i)+":0")
                    fd = self._create_feed_dict(model_in, keep=1.0)
                    if b in activity:
                        activity[b].append(h.eval(feed_dict=fd, session=self._session))
                    else:
                        activity[b] = [h.eval(feed_dict=fd, session=self._session)]
            return activity

    @property
    def convolution_data(self):
        """
        The weights and biases of the convolution layer(s)
        """
        self._check_init()
        with self._graph.as_default():
            g = self._session.graph
            w = {tg: g.get_tensor_by_name(self.cvn("WEIGHT", tg, -1)+":0").eval(session=self._session) for tg in ['t']}
            b = {tg: g.get_tensor_by_name(self.cvn("BIAS", tg, -1) + ":0").eval(session=self._session) for tg in ['t']}
        return w, b


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


class ModelData:
    """
    Provides access to model meta and checkpoint files by index from a given directory
    """
    def __init__(self, dirname):
        """
        Creates a new model-data instance
        :param dirname: Directory containing the model checkpoint and definition files
        """
        if not os.path.isdir(dirname):
            raise ValueError("dirname needs to be a valid directory")
        files = os.listdir(dirname)
        self.__meta_file = None
        self.__data_files = {}
        for f in files:
            if ".meta" in f:
                if self.__meta_file is None:
                    self.__meta_file = dirname + "/" + f
                else:
                    warn("Found at least two meta files in directory. Choosing first one.")
            elif ".data" in f:
                # obtain step index from filename and use as dictionary key
                num_start = f.find(".ckpt") + 6
                num_end = f.find(".data")
                try:
                    num = int(f[num_start:num_end])
                except ValueError:
                    continue
                self.__data_files[num] = dirname + "/" + f[:num_end]
        self.__input_dims = None
        self._hidden_sizes = None

    @property
    def CheckpointIndices(self):
        """
        Gets the indices of available checkpoints in ascending order
        """
        return sorted(self.__data_files.keys())

    @property
    def FirstCheckpoint(self):
        """
        Gets the filename of the first available model checkpoint (naive)
        """
        return self.__data_files[self.CheckpointIndices[0]]

    @property
    def LastCheckpoint(self):
        """
        Gets the filename of the last available model checkpoint (fully trained)
        """
        return self.__data_files[self.CheckpointIndices[-1]]

    @property
    def ModelDefinition(self):
        """
        Gets the filename of the model definition
        """
        return self.__meta_file

    def __contains__(self, item):
        return item in self.__data_files

    def __getitem__(self, item):
        if type(item) != int:
            raise TypeError("Key has to be integer")
        if item in self.__data_files:
            return self.__data_files[item]
        else:
            raise KeyError("No checkpoint with index {0} found.".format(item))


class GradientData:
    """
    Class that represents training/test data from a gradient experiment
    """
    def __init__(self, model_in, model_out, pred_window, frame_rate=GlobalDefs.frame_rate,
                 hist_seconds=GlobalDefs.hist_seconds):
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

    @property
    def standards(self):
        """
        The value standardizations
        """
        return GradientStandards(self.temp_mean, self.temp_std, self.disp_mean, self.disp_std, self.ang_mean,
                                 self.ang_std)

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
            grp2 = dfile.create_group("normalization")
            grp2.create_dataset("temp_mean", data=self.temp_mean)
            grp2.create_dataset("temp_std", data=self.temp_std)
            grp2.create_dataset("disp_mean", data=self.disp_mean)
            grp2.create_dataset("disp_std", data=self.disp_std)
            grp2.create_dataset("ang_mean", data=self.ang_mean)
            grp2.create_dataset("ang_std", data=self.ang_std)
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

    @staticmethod
    def load_standards(filename):
        """
        Loads training data standardizations from an hdf5 file
        :param filename: The file to load data from
        :return: A lightweight representation of only the standardization values
        """
        dfile = h5py.File(filename, 'r')
        if "model_in_raw" not in dfile or "model_out_raw" not in dfile:
            dfile.close()
            raise IOError("File does not seem to contain gradient data")
        if "normalization" not in dfile:
            dfile.close()
            raise IOError("File does not contain standardizations. Load full data instead.")
        grp = dfile["normalization"]
        return GradientStandards(np.array(grp["temp_mean"]), np.array(grp["temp_std"]), np.array(grp["disp_mean"]),
                                 np.array(grp["disp_std"]), np.array(grp["ang_mean"]), np.array(grp["ang_std"]))


class PersistentStore:
    """
    Base class for persistent storage of analysis results with an HDF5 backend
    """
    def __init__(self, db_file_name, read_only=False):
        """
        Create a new store with the indicated file as backend
        :param db_file_name: The path and name of backend file. File will be created if it doesn't exist. Can be None
            in which case store will exist non-persistent in memory.
        :param read_only: No changes allowed to backend and will fail if file does not exist
        """
        self._db_filename = db_file_name
        if db_file_name is None and read_only:
            raise ValueError("Pure memory storage can't be readonly")
        self._read_only = read_only
        self._db_file = None  # type: h5py.File

    def __enter__(self):
        """
        Entry point for context manager - open backend
        """
        if self._db_filename is None:
            # create memory-only representation for short-term caching
            import uuid
            self._db_file = h5py.File(str(uuid.uuid4()), "w", driver="core", backing_store=False)
            return self
        if self._read_only:
            self._db_file = h5py.File(self._db_filename, "r")
        else:
            self._db_file = h5py.File(self._db_filename, "a")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit context manager, close file backend
        """
        if self._db_file is not None:
            self._db_file.close()
        self._db_file = None

    def _check_open(self):
        """
        Checks whether the database file has been opened raises NotInitialized otherwise
        """
        if self._db_file is None:
            raise NotInitialized("Backend is not open")

    @staticmethod
    def _identifier(*args):
        return '/'.join([str(a) for a in args])

    def _get_data(self, *args):
        """
        Concatenates arguments into hdf5 path and returns data if it exists
        :param args: Ordered series of group/data keys
        :return: The data identified by the arguments or None if it does not exist
        """
        self._check_open()
        identifier = self._identifier(*args)
        if identifier in self._db_file:
            return np.array(self._db_file[identifier])
        else:
            return None

    def _set_data(self, data: np.ndarray, *args):
        """
        Sets/updates data at hdf5 path indicated by *args with given data
        :param data: The data to store
        :param args: Ordered series of group/data keys
        """
        self._check_open()
        if self._read_only:
            raise IOError("Setting not allowed on readonly object")
        identifier = self._identifier(*args)
        if identifier in self._db_file:
            del self._db_file[identifier]
        try:
            self._db_file.create_dataset(identifier, data=data, compression="gzip", compression_opts=6)
        except TypeError:
            # data can't be compressed - the case for pickle strings wrapped as np.void
            self._db_file.create_dataset(identifier, data=data)

    def _clear_data(self, *args):
        """
        Clears data at hdf5 path indicated by *args. Raises KeyError if data does not exist
        :param args: Ordered series of group/data keys
        """
        self._check_open()
        if self._read_only:
            raise IOError("Deleting not allowed on readonly object")
        identifier = self._identifier(*args)
        if identifier in self._db_file:
            del self._db_file[identifier]
        else:
            raise KeyError("Could not find data identifier in file")
