from __future__ import division
import tensorflow as tf
import numpy as np
import gzip
import os
import sys
import tensorflow.contrib.slim as slim
from random import shuffle

# Show all variables in current model
def show_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    
# Create folders if they do not already exist
def checkFolders(dir_list):
    for dir in list(dir_list):
        if not os.path.exists(dir):
            os.makedirs(dir)

# Check that fulle MNIST dataset exists in specified directory
def checkData(data_dir):
    if not os.path.exists(data_dir):
        raise OSError("Specified data directory '" + data_dir + "' does not exist in filesystem.")
    elif not os.path.exists(os.path.join(data_dir,'t10k-images-idx3-ubyte.gz')):
        raise OSError("'t10k-images-idx3-ubyte.gz' not found in data directory.")
    elif not os.path.exists(os.path.join(data_dir,'train-images-idx3-ubyte.gz')):
        raise OSError("'train-images-idx3-ubyte.gz' not found in data directory.")
    elif not os.path.exists(os.path.join(data_dir,'t10k-labels-idx1-ubyte.gz')):
        raise OSError("'t10k-labels-idx1-ubyte.gz' not found in data directory.")
    elif not os.path.exists(os.path.join(data_dir,'train-labels-idx1-ubyte.gz')):
        raise OSError("'train-labels-idx1-ubyte.gz' not found in data directory.")

            
# Add suffix to end of tensor name
def add_suffix(name, suffix):
    if suffix is not None:
        return name + suffix
    else:
        return name
        
# Creates byte feature for storing numpy arrays        
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Define loader for MNIST dataset
# (available at http://yann.lecun.com/exdb/mnist/)
def write_mnist_tfrecords():
    # The four .gz files for the MNIST dataset
    # should be placed in the "./data/" directory
    data_dir = "./data/"

    """
    MNIST Loader modified from utils file in GitHub repo by 'hwalsuklee':
    https://github.com/hwalsuklee/tensorflow-generative-model-collections
    """

    # Function for extracting data from gunzipped files
    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    # Load and merge handwritten digits from training and test datasets
    x_data_1 = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    x_data_2 = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    x_data = np.concatenate((np.reshape(x_data_1,(60000, 28, 28, 1)),
                             np.reshape(x_data_2,(10000, 28, 28, 1))), axis=0)

    # Load and merge digit labels for training and test datasets
    y_data_1 = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    y_data_2 = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    y_data = np.concatenate((np.reshape(y_data_1,(60000)),
                             np.reshape(y_data_2,(10000))), axis=0).astype(np.int)

    # Convert labels to collection of one-hot vectors
    y_onehot = np.zeros((len(y_data), 10), dtype=np.uint8)
    for n in range(0,len(y_data)):
        y_onehot[n, y_data[n]] = 1

    # Ensure all data is stored in uint8 format
    x_data = x_data.astype(np.uint8)
    y_onehot = y_onehot.astype(np.uint8)

    # Shuffle data and create training and validation sets
    data_count = x_data.shape[0]
    indices = [n for n in range(0,data_count)]
    shuffle(indices)
    t_indices = indices[0 : int(np.floor(0.8 * data_count))]
    v_indices = indices[int(np.floor(0.8 * data_count)) : ]

    """
    Reference for tfrecords writing and reading:
    http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html
    """

    # Save training dataset in .tfrecords file
    train_filename = './data/training.tfrecords'
    writer = tf.python_io.TFRecordWriter(train_filename)
    for i in t_indices:
        img = x_data[i]
        label = y_onehot[i]

        # Create a feature
        feature = {'image': _bytes_feature(img.tostring()),
                   'label': _bytes_feature(label.tostring())}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write to file
        writer.write(example.SerializeToString())

    # Close .tfrecords writer
    writer.close()

    # Save validation dataset in .tfrecords file
    val_filename = './data/validation.tfrecords'
    writer = tf.python_io.TFRecordWriter(val_filename)
    for i in v_indices:
        img = x_data[i]
        label = y_onehot[i]

        # Create a feature
        feature = {'image': _bytes_feature(img.tostring()),
                   'label': _bytes_feature(label.tostring())}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write to file
        writer.write(example.SerializeToString())
        
    # Close .tfrecords writer            
    writer.close()

# Read one example from tfrecords file (used to check formatting)
def read_mnist_tfrecords():
    reader = tf.TFRecordReader()
    filenames = './data/training.tfrecords'
    filename_queue = tf.train.string_input_producer(filenames)
    _, serialized_example = reader.read(filename_queue)
    feature_set = {'image': tf.FixedLenFeature([], tf.string),
                   'label': tf.FixedLenFeature([], tf.string)}
    features = tf.parse_single_example( serialized_example, features= feature_set )
    raw_image = features['image']
    image = tf.decode_raw(raw_image, tf.uint8)
    image = tf.reshape(image, [28, 28, 1])
    raw_label = features['label']
    label = tf.decode_raw(raw_label, tf.uint8)

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        img, lab = sess.run([image,label])
        print([img, lab])

