import tensorflow as tf
import numpy as np
import time
from random import shuffle

# Import base model for defining early stopping hook
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/session_run_hook.py
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util


# Transforms 'example_proto' byte strings into decoded
# onehot label and resized image array (only returns the image)
def _parse_mnist_image(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.string, default_value="")}
    parsed_features = tf.parse_single_example(example_proto, features)
    image = tf.decode_raw(parsed_features["image"], tf.uint8)
    image = tf.cast(tf.reshape(image, [28, 28, 1]), tf.float32)
    image = tf.divide(image, tf.constant(255.))
    return image

# Transforms 'example_proto' byte strings into decoded
# onehot label and resized image array 
def _parse_mnist_data(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.string, default_value="")}
    parsed_features = tf.parse_single_example(example_proto, features)
    image = tf.decode_raw(parsed_features["image"], tf.uint8)
    image = tf.cast(tf.reshape(image, [28, 28, 1]), tf.float32)
    image = tf.divide(image, tf.constant(255.))
    label = tf.decode_raw(parsed_features["label"], tf.uint8)
    label = tf.cast(label, tf.float32)
    return image, label


# Define early stopping hook
class EarlyStoppingHook(session_run_hook.SessionRunHook):
    def __init__(self, loss_name, feed_dict={}, tolerance=0.01, stopping_step=50, start_step=100):
        self.loss_name = loss_name
        self.feed_dict = feed_dict
        self.tolerance = tolerance
        self.stopping_step = stopping_step
        self.start_step = start_step

    # Initialize global and internal step counts
    def begin(self):
        self._global_step_tensor = training_util._get_or_create_global_step_read()
        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use EarlyStoppingHook.")
        self._prev_step = -1
        self._step = 0

    # Evaluate early stopping loss every 1000 steps
    # (avoiding repetition when multiple run calls are made each step)
    def before_run(self, run_context):
        if (self._step % self.stopping_step == 0) and \
           (not self._step == self._prev_step) and (self._step > self.start_step):

            print("\n[ Early Stopping Check ]")
            
            # Get graph from run_context session
            graph = run_context.session.graph

            # Retrieve loss tensor from graph
            loss_tensor = graph.get_tensor_by_name(self.loss_name)

            # Populate feed dictionary with placeholders and values
            fd = {}
            for key, value in self.feed_dict.items():
                placeholder = graph.get_tensor_by_name(key)
                fd[placeholder] = value

            return session_run_hook.SessionRunArgs({'step': self._global_step_tensor,
                                                    'loss': loss_tensor}, feed_dict=fd)
        else:
            return session_run_hook.SessionRunArgs({'step': self._global_step_tensor})
                                                    
    # Check if current loss is below tolerance for early stopping
    def after_run(self, run_context, run_values):
        if (self._step % self.stopping_step == 0) and \
           (not self._step == self._prev_step) and (self._step > self.start_step):
            global_step = run_values.results['step']
            current_loss = run_values.results['loss']
            print("Current stopping loss  =  %.10f\n" %(current_loss))
            
            if current_loss < self.tolerance:
                print("[ Early Stopping Criterion Satisfied ]\n")
                run_context.request_stop()
            self._prev_step = global_step            
        else:
            global_step = run_values.results['step']
            self._step = global_step

