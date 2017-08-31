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


# Constants
FRAME_RATE = 100  # all simulations, training data models have a native framerate of 100 Hz
HIST_SECONDS = 4  # all inputs to the models are 4s into the past
MODEL_RATE = 5    # model input rate is 5 Hz
PRED_WINDOW = int(FRAME_RATE * 0.5)  # the model should predict the temperature 500 ms into the future


# Functions
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


def hidden_temperature_responses(model_file: str, chkpoint: str, n_hidden, t_stimulus, t_mean, t_std):
    """
    Computes and returns the responses of each hidden unit in the network in response to a temperature stimulus
    (all other inputs are clamped to 0)
    :param model_file: The file of the model definitions (.meta)
    :param chkpoint: The model checkpoint (.ckpt)
    :param n_hidden: The number of hidden layers in the network
    :param t_stimulus: The temperature stimulus in C
    :param t_mean: The average temperature of the stimulus used to train the network
    :param t_std: The temperature standard deviation of the stimulus used to train the network
    :return: List with n_hidden elements, each an array of time x n_units activations of hidden units
    """
    # TODO: Remove explicit n_hidden parameter and instead retrieve somehow from model
    history = FRAME_RATE * HIST_SECONDS  # TODO: This should come from the model definition x_in
    activity = []
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_file)
        saver.restore(sess, chkpoint)
        graph = tf.get_default_graph()
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        x_in = graph.get_tensor_by_name("x_in:0")
        ix = indexing_matrix(np.arange(t_stimulus.size), history-1, 0, t_stimulus.size)[0]
        model_in = np.zeros((ix.shape[0], 3, history, 1))
        model_in[:, 0, :, 0] = (t_stimulus[ix] - t_mean) / t_std
        for i in range(n_hidden):
            h = graph.get_tensor_by_name("h_{0}:0".format(i))
            activity.append(h.eval(feed_dict={x_in: model_in, keep_prob: 1.0}))
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
