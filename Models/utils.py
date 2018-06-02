"""
MNIST Loader modified from utils file in GitHub repo by 'hwalsuklee':
https://github.com/hwalsuklee/tensorflow-generative-model-collections
"""
from __future__ import division
import numpy as np
import gzip

# Define loader for MNIST dataset
# (available at http://yann.lecun.com/exdb/mnist/)
def load_mnist():

    # The four .gz files for the MNIST dataset
    # should be placed in the "./data" directory
    data_dir = "./data/"

    # Extract function for reading gunzipped data
    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    # Load and merge handwritten digits from train and test datasets
    x_data_1 = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    x_data_2 = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    x_data = np.concatenate((np.reshape(x_data_1,(60000, 28, 28, 1)),
                             np.reshape(x_data_2,(10000, 28, 28, 1))), axis=0)

    # Load and merge digit labels for train and test datasets
    y_data_1 = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    y_data_2 = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    y_data = np.concatenate((np.reshape(y_data_1,(60000)),
                             np.reshape(y_data_2,(10000))), axis=0).astype(np.int)

    # Convert labels to collection of one-hot vectors
    y_onehot = np.zeros((len(y_data), 10), dtype=np.float)
    for n in range(0,len(y_data)):
        y_onehot[n, y_data[n]] = 1.0

    # Rescale pixels
    x_data = x_data/255.

    return x_data, y_onehot
