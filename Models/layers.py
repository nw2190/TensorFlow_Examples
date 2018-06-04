import tensorflow as tf
import numpy as np

# Define default weight initializer for all layers
default_wt_init = tf.truncated_normal_initializer(stddev=0.02)


# Define dense/fully-connected layer
def dense(x, nodes, activation=None, wt_init=default_wt_init, reuse=None, name=None):
    y = tf.layers.dense(x, nodes, activation=activation, kernel_initializer=wt_init, reuse=reuse, name=name)
    return y

# Define convolutional layer for two-dimensional data
def conv2d(x, channels, kernel_size=4, strides=2, activation=None, wt_init=default_wt_init, reuse=None, name=None):
    y = tf.layers.conv2d(x, channels, kernel_size,
                         strides=strides, padding='same', activation=activation,
                         kernel_initializer=wt_init, reuse=reuse, name=name)
    return y

# Define convolutional transpose layer for two-dimensional data
def conv2d_transpose(x, channels, kernel_size=4, strides=2, activation=None, wt_init=default_wt_init, reuse=None, name=None):
    y = tf.layers.conv2d_transpose(x, channels, kernel_size,
                                   strides=strides, padding='same', activation=activation,
                                   kernel_initializer=wt_init, reuse=reuse, name=name)
    return y

# Define batch normalization layer
def batch_norm(x, training=True, reuse=None, name=None):
    y = tf.layers.batch_normalization(x, scale=True, momentum=0.9, epsilon=1e-5,
                                      training=training, reuse=reuse, name=name)
    return y
