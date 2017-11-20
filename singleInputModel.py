# Copyright 2017 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Builds a gradient prediction network that uses separated input layers, i.e. the input is a 2D tensor of both behavior
and temperature variables on which convolution filters operate separately feeding into separated hidden layers before
merging
"""

import tensorflow as tf
import core
import numpy as np


class NotInitialized(Exception):
    def __init__(self, message):
        super.__init__(message)


class GpNetworkModel:
    """
    Class representing gradient prediction network models
    """
    def __init__(self, n_conv_layers: int, n_units: int, n_layers_branch: int, n_layers_mixed: int):
        """
        Creates a new GpNetworkModel with the indicated structure. Does not initialize the graph itself.
        :param n_conv_layers: The number of convolutional layers per input branch
        :param n_units: The number of units in each hidden layer
        :param n_layers_branch: The number of hidden layers in each branch (can be 0 for full mixing)
        :param n_layers_mixed: The number of hidden layers in the mixed part of the model
        """
        tf.reset_default_graph()
        self.initialized = False
        # ingest parameters
        if n_layers_mixed < 1:
            raise ValueError("Network needs at least on mixed hidden layer")
        if n_layers_branch < 0:
            raise ValueError("Number of branch layers can't be negative")
        self.n_conv_layers = n_conv_layers
        self.n_units = n_units
        self.n_layers_branch = n_layers_branch
        self.n_layers_mixed = n_layers_mixed
        # set training defaults
        self.w_decay = 1e-4
        self.keep_train = 0.5
        assert core.FRAME_RATE % core.MODEL_RATE == 0
        self.t_bin = core.FRAME_RATE // core.MODEL_RATE  # bin input down to 5Hz
        self.binned_size = core.FRAME_RATE * core.HIST_SECONDS // self.t_bin
        # initialize fields that will be populated later
        self._n_mixed_dense = None
        self._n_branch_dense = None
        self._branches = None
        self._keep_prob = None
        self._det_remove = {}

    # Private API
    def _create_unit_lists(self):
        """
        Creates lists of hidden unit counts and branch list according to network configuration
        """
        self._det_remove = {}
        self._n_mixed_dense = [self.n_units] * self.n_layers_mixed
        if self.n_layers_branch == 0:
            self._branches = ['m', 'o']  # mixed network
        else:
            self._branches = ['t', 's', 'a', 'm', 'o']  # single input network
            self._n_branch_dense = [self.n_units] * self.n_layers_mixed

    def _check_init(self):
        """
        Checks if network is initialized and raises exception otherwise
        """
        if not self.initialized:
            raise NotInitialized("Can't perform operation before performing setup of graph.")

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
        w = core.create_weight_var(self.cvn("WEIGHT", branch, index), [prev_out.shape[1].value, n_units], self.w_decay)
        b = core.create_bias_var(self.cvn("BIAS", branch, index), [n_units])
        dr = self._det_remove[branch][index]
        scale = n_units / tf.reduce_sum(dr)
        h = tf.nn.relu((tf.matmul(prev_out, w) + b) * dr * scale, self.cvn("HIDDEN", branch, index))
        return h

    def _create_output(self, prev_out: tf.Tensor) -> tf.Tensor:
        """
        Creates the output layer for reporting predicted temperature of all four behaviors
        :param prev_out: The output of the previous layer
        :return: output
        """
        w = core.create_weight_var(self.cvn("WEIGHT", 'o', 0), [prev_out.shape[1].value, 4], self.w_decay)
        b = core.create_bias_var(self.cvn("BIAS", 'o', 0), [4])
        out = tf.add(tf.matmul(prev_out, w), b, name=self.cvn("OUTPUT", 'o', 0))
        return out

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

    # Public API
    def setup(self):
        """
        Creates the network graph
        """
        self.clear()
        self._create_unit_lists()
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
        # TODO: Create the full network graph
        # mark network as initialized
        self.initialized = True

    def clear(self):
        """
        Clears the network graph
        """
        if not self.initialized:
            return
        tf.reset_default_graph()
        # mark network as not initialized
        self.initialized = False

    def cvn(self, vartype: str, branch: str, index: int) -> str:
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
        if branch not in self._branches:
            raise ValueError("Unknown branch type {0}. Has to be in {1}".format(branch, self._branches))
        return "{0}_{1}_{2}".format(vartype, branch, index)


if __name__ == "__main__":
    import matplotlib.pyplot as pl
    import seaborn as sns
    print("Testing separate input model.", flush=True)
