#  Copyright 2017 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Commonly used helper functions
"""

import tensorflow as tf


def create_weight_var(name, shape, w_decay=None, loss_collection="losses"):
    """
    Creates a weight variable with optional weight decay initialized with sd = 1/size
    :param name: The name of the variable
    :param shape: The desired shape
    :param w_decay: None or L2 loss term if weight decay is desired
    :param loss_collection: The name of the collection to which loss should be added
    :return: The weight variable
    """
    initial = tf.truncated_normal(shape, stddev=1.0 / sum(shape))
    var = tf.Variable(initial, name=name)
    if w_decay is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), w_decay, name="l2_w_loss")
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
