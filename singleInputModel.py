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
        super().__init__(message)


class GpNetworkModel:
    """
    Class representing gradient prediction network models
    """
    def __init__(self):
        """
        Creates a new GpNetworkModel
        """
        tf.reset_default_graph()
        self.initialized = False
        # set training defaults
        self.w_decay = 1e-4
        self.keep_train = 0.5
        assert core.FRAME_RATE % core.MODEL_RATE == 0
        self.t_bin = core.FRAME_RATE // core.MODEL_RATE  # bin input down to 5Hz
        self.binned_size = core.FRAME_RATE * core.HIST_SECONDS // self.t_bin
        # initialize fields that will be populated later
        self.n_conv_layers = None
        self.n_units = None
        self.n_layers_branch = None
        self.n_layers_mixed = None
        self._n_mixed_dense = None
        self._n_branch_dense = None
        self._branches = None
        # our session object
        self._session = None  # type: tf.Session
        # model fields that are later needed for parameter feeding
        self._keep_prob = None  # drop-out keep probability
        self._det_remove = {}  # vectors for deterministic keeping/removal of individual units
        self._x_in = None  # network inputs
        self._y_ = None  # true responses (for training)
        # network output
        self._m_out = None  # type: tf.Tensor
        # the square loss (loss w.o. weight decay)
        self._sq_loss = None  # type: tf.Tensor
        # total loss across the network
        self._total_loss = None  # type: tf.Tensor
        # the training step to train the network
        self._train_step = None  # type: tf.Operation
        # saver object to save progress
        self._saver = None  # type: tf.train.Saver

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear()
        return False  # re-raise any previous exceptions

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
        w_conv1 = core.create_weight_var(self.cvn("WEIGHT", branch, -1), [len_0, self.binned_size, 1, self.n_conv_layers])
        b_conv1 = core.create_bias_var(self.cvn("BIAS", branch, -1), [self.n_conv_layers])
        conv1 = core.create_conv2d(self.cvn("CONV", branch, -1), prev_out, w_conv1)
        cname = self.cvn("HIDDEN", branch, -1)
        h_conv1 = tf.nn.relu(conv1 + b_conv1, cname)
        h_conv1_flat = tf.reshape(h_conv1, [-1, self.n_conv_layers], cname+"_flat")
        return h_conv1_flat

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
                if removal is None:
                    f_dict[dr] = np.ones(s, dtype=np.float32)
                else:
                    if removal[b][i].size != s:
                        raise ValueError("removal in branch {0} layer {1} does not have required size of {2}".format(b,
                                                                                                                     i,
                                                                                                                     s))
                    f_dict[dr] = removal[b][i]
        return f_dict

    # Public API
    def setup(self, n_conv_layers: int, n_units: int, n_layers_branch: int, n_layers_mixed: int):
        """
        Creates the network graph from scratch according to the given specifications
        :param n_conv_layers: The number of convolutional layers per input branch
        :param n_units: The number of units in each hidden layer
        :param n_layers_branch: The number of hidden layers in each branch (can be 0 for full mixing)
        :param n_layers_mixed: The number of hidden layers in the mixed part of the model
        """
        # ingest parameters
        if n_layers_mixed < 1:
            raise ValueError("Network needs at least on mixed hidden layer")
        if n_layers_branch < 0:
            raise ValueError("Number of branch layers can't be negative")
        self.n_conv_layers = n_conv_layers
        self.n_units = n_units
        self.n_layers_branch = n_layers_branch
        self.n_layers_mixed = n_layers_mixed
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
        # model input: BATCHSIZE x (Temp,Move,Turn) x HISTORYSIZE x 1 CHANNEL
        self._x_in = tf.placeholder(tf.float32, [None, 3, core.FRAME_RATE * core.HIST_SECONDS, 1], "x_in")
        # real outputs: BATCHSIZE x (dT(Stay), dT(Straight), dT(Left), dT(Right))
        self._y_ = tf.placeholder(tf.float32, [None, 4], "y_")
        # data binning layer
        xin_pool = core.create_meanpool2d("xin_pool", self._x_in, 1, self.t_bin)
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
        self._total_loss, self._sq_loss = core.get_loss(self._y_, self._m_out)
        self._train_step = core.create_train_step(self._total_loss)
        # create session
        self._session = tf.Session()
        # mark network as initialized
        self.initialized = True
        # intialize all variables
        self.init_variables()

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
        self._saver = None
        tf.reset_default_graph()
        self.n_conv_layers = None
        self.n_units = None
        self.n_layers_branch = None
        self.n_layers_mixed = None
        # mark network as not initialized
        self.initialized = False

    def init_variables(self):
        """
        Runs global variable initializer, resetting all values
        """
        self._check_init()
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
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=None)  # never delete chkpoint files
        return self._saver.save(self._session, chkpoint_file, global_step=index, write_meta_graph=save_meta)

    def train(self, xbatch, ybatch, keep=0.5):
        """
        Runs a training step on the given batches
        :param xbatch: The input of the training batch
        :param ybatch: The true labels of the training batch
        :param keep: The keep probability of each unit
        """
        self._check_init()
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
        return self._sq_loss.eval(self._create_feed_dict(xbatch, ybatch, keep), self._session)

    def predict(self, xbatch, keep=1, det_drop=None) -> tf.Tensor:
        """
        Uses the network to predict output given the input
        :param xbatch: The network input
        :param keep: The keep probability of each unit
        :param det_drop: The deterministic keep/drop of each unit
        :return: The network output
        """
        self._check_init()
        return self._m_out.eval(self._create_feed_dict(xbatch, keep=keep, removal=det_drop), session=self._session)

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

    @property
    def convolution_data(self):
        """
        The weights and biases of the convolution layer(s)
        """
        self._check_init()
        if 't' in self._branches:
            to_get = ['t', 's', 'a']
        else:
            to_get = ['m']
        g = self._session.graph
        w = {tg: g.get_tensor_by_name(self.cvn("WEIGHT", tg, -1)+":0").eval(session=self._session) for tg in to_get}
        b = {tg: g.get_tensor_by_name(self.cvn("BIAS", tg, -1) + ":0").eval(session=self._session) for tg in to_get}
        return w, b


if __name__ == "__main__":
    N_CONV_LAYERS = 10
    import matplotlib.pyplot as pl
    import seaborn as sns
    from scipy.ndimage import gaussian_filter1d
    print("Testing separate input model.", flush=True)
    print("For each 'behavior' subpart attempt to learn different sums on standard normal distribution", flush=True)
    t_losses = []
    d_fracs = []
    with GpNetworkModel() as model:
        model.setup(N_CONV_LAYERS, 512, 2, 2)
        for i in range(2000):
            xb1 = np.random.randn(100, 1, core.FRAME_RATE * core.HIST_SECONDS, 1)
            xb2 = xb1 ** 2
            xb2 -= 1  # expected average of xb1**2
            xb3 = xb1 ** 3
            xbatch = np.concatenate((xb1, xb2, xb3), 1)
            ybatch = np.c_[np.sum(xb2, axis=(1, 2)), np.sum(xb2 / 4, axis=(1, 2)),
                           np.sum(xb1, axis=(1, 2)), np.sum(xb1 / 2, axis=(1, 2))]
            cur_l = model.get_squared_loss(xbatch, ybatch)
            pred = model.predict(xbatch)
            cur_d = np.median(np.abs((ybatch - pred) / ybatch))
            t_losses.append(cur_l)
            d_fracs.append(cur_d)
            if i % 200 == 0:
                print('step %d, training loss %g, delta fraction %g' % (i, cur_l, cur_d))
            model.train(xbatch, ybatch)
        weights_conv, bias_conv = model.convolution_data

    weights_conv = weights_conv['t']
    w_ext = np.max(np.abs(weights_conv))
    fig, ax = pl.subplots(ncols=int(np.sqrt(N_CONV_LAYERS)), nrows=int(np.sqrt(N_CONV_LAYERS)), frameon=False,
                          figsize=(14, 2.8))
    ax = ax.ravel()
    for i, a in enumerate(ax):
        sns.heatmap(weights_conv[:, :, 0, i], ax=a, vmin=-w_ext, vmax=w_ext, center=0, cbar=False)
        a.axis("off")

    pl.figure()
    pl.plot(t_losses, 'o')
    pl.plot(gaussian_filter1d(t_losses, 25))
    pl.xlabel("Batch")
    pl.ylabel("Training loss")
    sns.despine()

    pl.figure()
    pl.plot(d_fracs, 'o')
    pl.plot(gaussian_filter1d(d_fracs, 25))
    pl.xlabel("Batch")
    pl.ylabel("Error fraction")
    sns.despine()
