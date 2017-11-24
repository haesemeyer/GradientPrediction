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


# Constants
FRAME_RATE = 100  # all simulations, training data models have a native framerate of 100 Hz
HIST_SECONDS = 4  # all inputs to the models are 4s into the past
MODEL_RATE = 5    # model input rate is 5 Hz
PRED_WINDOW = int(FRAME_RATE * 0.5)  # the model should predict the temperature 500 ms into the future


# Functions
def ca_convolve(trace, ca_timeconstant, frame_rate):
    """
    Convolves a trace with a decaying calcium kernel
    :param trace: The activity trace to convolve
    :param ca_timeconstant: The timeconstant of the calcium indicator
    :param frame_rate: The original frame-rate to relate samples to the time constant
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

    if ca_timeconstant == 0:
        return trace
    kernel = ca_kernel(ca_timeconstant, frame_rate)
    return np.convolve(trace, kernel)[:trace.size]


def create_weight_var(name, shape, w_decay=None, loss_collection="losses"):
    """
    Creates a weight variable with optional weight decay initialized with sd = 1/size
    :param name: The name of the variable
    :param shape: The desired shape
    :param w_decay: None or L2 loss term if weight decay is desired
    :param loss_collection: The name of the collection to which loss should be added
    :return: The weight variable
    """
    var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
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


def test_error_distributions(model_file: str, chkpoint: str, test_data):
    """
    For a given model at a given checkpoint returns the distribution of squared losses and rank errors across test data
    :param model_file: The file of the model definitions (.meta)
    :param chkpoint: The model checkpoint (.ckpt)
    :param test_data: The test data to evaluate the model
        [0]: For each datapoint in test_data the squared error loss
        [1]: For each datapoint in test_data the rank error
    """
    sq_errors = np.full(test_data.data_size, -1)
    rank_errors = np.full(test_data.data_size, -1)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_file)
        saver.restore(sess, chkpoint)
        graph = tf.get_default_graph()
        m_out = graph.get_tensor_by_name("m_out:0")
        x_in = graph.get_tensor_by_name("x_in:0")
        y_ = graph.get_tensor_by_name("y_:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        for i in range(test_data.data_size):
            xbatch, ybatch = test_data.training_batch(1)
            pred = m_out.eval(feed_dict={x_in: xbatch, y_: ybatch, keep_prob: 1.0})
            sq_errors[i] = (np.sum((ybatch - pred)**2))
            rank_real = np.unique(ybatch, return_inverse=True)[1]
            rank_pred = np.unique(pred, return_inverse=True)[1]
            rank_errors[i] = np.sum(np.abs(rank_real - rank_pred))
    return sq_errors, rank_errors


def hidden_temperature_responses(model, chkpoint, t_stimulus, t_mean, t_std):
    """
    Computes and returns the responses of each hidden unit in the network in response to a temperature stimulus
    (all other inputs are clamped to 0)
    :param model: ModelData object to describe model
    :param chkpoint: The model checkpoint (.ckpt) file or index into ModelData
    :param t_stimulus: The temperature stimulus in C
    :param t_mean: The average temperature of the stimulus used to train the network
    :param t_std: The temperature standard deviation of the stimulus used to train the network
    :return: List with n_hidden elements, each an array of time x n_units activations of hidden units
    """
    # if chkpoint is not a string we assume it is meant as index into model - validate
    if type(chkpoint) != str:
        if chkpoint not in model:
            raise ValueError("{0} is not a valid model checkpoint".format(chkpoint))
        else:
            chkpoint = model[chkpoint]
    n_hidden = model.get_n_hidden()
    history = model.get_input_dims()[2]
    activity = []
    tf.reset_default_graph()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model.ModelDefinition)
        saver.restore(sess, chkpoint)
        graph = tf.get_default_graph()
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        x_in = graph.get_tensor_by_name("x_in:0")
        ix = indexing_matrix(np.arange(t_stimulus.size), history-1, 0, t_stimulus.size)[0]
        model_in = np.zeros((ix.shape[0], 3, history, 1))
        model_in[:, 0, :, 0] = (t_stimulus[ix] - t_mean) / t_std
        for i in range(n_hidden):
            h = graph.get_tensor_by_name("h_{0}:0".format(i))
            fd = {x_in: model_in, keep_prob: 1.0}
            # Add det remove to feed dict where appropriate
            try:
                for j in range(n_hidden):
                    dr = graph.get_tensor_by_name("remove_{0}:0".format(j))
                    fd[dr] = np.ones(dr.shape.as_list()[0])
            except KeyError:
                pass
            activity.append(h.eval(feed_dict=fd))
    return activity


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

    def get_input_dims(self):
        """
        Returns the number of history frames the network expects in the input
        """
        if self.__input_dims is None:
            # first check if a session is already open otherwise open new one
            if tf.get_default_session() is None:
                tf.reset_default_graph()
                with tf.Session():
                    tf.train.import_meta_graph(self.ModelDefinition)
                    graph = tf.get_default_graph()
                    x_in = graph.get_tensor_by_name("x_in:0")
                    self.__input_dims = x_in.shape.as_list()
            else:
                graph = tf.get_default_graph()
                x_in = graph.get_tensor_by_name("x_in:0")
                self.__input_dims = x_in.shape.as_list()
        return self.__input_dims.copy()

    def get_n_hidden(self):
        """
        Returns the number of hidden layers in the model
        """
        return len(self.get_hidden_sizes())

    def get_hidden_sizes(self):
        """
        Returns the number of units in each hidden layer of the network
        """
        if self._hidden_sizes is None:
            # first check if a session is already open otherwise open new one
            if tf.get_default_session() is None:
                tf.reset_default_graph()
                with tf.Session():
                    tf.train.import_meta_graph(self.ModelDefinition)
                    graph = tf.get_default_graph()
                    # we use the same strategy as above to identify the operations belonging to our hidden layers
                    # then we access the corresponding tensor and it's shape in position 1 will be the number of hidden
                    # units
                    self._hidden_sizes = [graph.get_tensor_by_name(op.name+":0").shape.as_list()[1] for op in
                                          graph.get_operations() if op.type == "Relu" and len(op.name) == 3 and "h_" in
                                          op.name]
            else:
                graph = tf.get_default_graph()
                self._hidden_sizes = [graph.get_tensor_by_name(op.name + ":0").shape.as_list()[1] for op in
                                      graph.get_operations() if op.type == "Relu" and len(op.name) == 3 and "h_" in
                                      op.name]
        return self._hidden_sizes

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


class TemperatureArena:
    """
    Base class for behavioral simulations that happen in a
    virtual arena where temperature depends on position
    """
    def __init__(self):
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


class ModelSimulation(TemperatureArena):
    """
    Base class for simulations that use trained networks
    to perform gradient navigation
    """
    def __init__(self, model: ModelData, chkpoint, tdata, t_preferred):
        """
        Creates a new ModelSimulation
        :param model: The ModelData describing our network model
        :param chkpoint: The desired checkpoint file to use for the simulation
        :param tdata: Training data or related object to supply scaling information
        :param t_preferred: The preferred temperature that should be reached during the simulation
        """
        super().__init__()
        self.model = model
        self.chkpoint = chkpoint
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

    @property
    def model_file(self):
        return self.model.ModelDefinition

    def create_feed_dict(self, x_in, xvals, det_remove, keep_prob):
        """
        Creates a feeding dictionary for our model
        :param x_in: Model variable of model inputs
        :param xvals: The actual input values
        :param det_remove: List of model tensors for deterministic removal
        :param keep_prob: Model tensor of dropout probability
        :return: The feeding dictionary for this model interation
        """
        fd = {x_in: xvals, keep_prob: 1.0}
        if det_remove is None:
            return fd
        if self.remove is None:
            for dr in det_remove:
                fd[dr] = np.ones(dr.shape.as_list()[0])
        else:
            if len(det_remove) != len(self.remove):
                raise ValueError("self.remove has a different amount of elements than hidden network layers")
            for i, dr in enumerate(det_remove):
                if self.remove[i].size == dr.shape.as_list()[0]:
                    fd[dr] = self.remove[i]
                else:
                    raise ValueError("All elements of self.remove need to comply with hidden layer sizes")
        return fd

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
        history = FRAME_RATE * HIST_SECONDS
        burn_period = history * 2
        start = history + 1
        pos = np.full((nsteps + burn_period, 3), np.nan)
        pos[:start + 1, :] = self.get_start_pos()[None, :]
        # start session, load model and run simulation
        tf.reset_default_graph()
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(self.model_file)
            saver.restore(sess, self.chkpoint)
            graph = tf.get_default_graph()
            m_out = graph.get_tensor_by_name("m_out:0")
            x_in = graph.get_tensor_by_name("x_in:0")
            keep_prob = graph.get_tensor_by_name("keep_prob:0")
            # obtain list of deterministic removal placeholders
            n_hidden = self.model.get_n_hidden()
            try:
                det_remove = [graph.get_tensor_by_name("remove_{0}:0".format(i)) for i in range(n_hidden)]
            except KeyError:
                # this model was saved before adding deterministic removal of units
                det_remove = None
            # start simulation
            step = start
            model_in = np.zeros((1, 3, history, 1))
            # overall bout frequency at ~1 Hz
            p_eval = 1.0 / FRAME_RATE
            while step < nsteps + burn_period:
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
                fd = self.create_feed_dict(x_in, model_in, det_remove, keep_prob)
                model_out = m_out.eval(feed_dict=fd).ravel()
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
        history = FRAME_RATE * HIST_SECONDS
        burn_period = history * 2
        start = history + 1
        pos = np.full((nsteps + burn_period, 3), np.nan)
        pos[:start + 1, :] = self.get_start_pos()[None, :]
        step = start
        # overall bout frequency at ~1 Hz
        p_eval = 1.0 / FRAME_RATE
        t_out = np.zeros(4)
        while step < nsteps + burn_period:
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
