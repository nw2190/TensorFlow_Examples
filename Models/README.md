# Models
Collection of TensorFlow Models for working with the MNIST dataset of handwritten digits.

### Dataset
The MNIST dataset of handwritten digits is available at:
* [MNIST](http://yann.lecun.com/exdb/mnist/) - http://yann.lecun.com/exdb/mnist/
    
The four gunzipped files should be placed the subdirectory './data/' of the 'Models/' folder.  The first time a model is called, the 'write_mnist_tfrecords()' function from the 'utils.py' file is executed in order to create the 'training.tfrecords' and 'validation.tfrecords' files used for training.  

### Running models
Run options for each model are definied in the flags.py file.  Default options are specified for each argument for convenience; to override a default option simply run, e.g.:
```
$ python 01_Classifier.py  --batch_size 100
```

Model checkpoints, logs, and predictions are stored in the './Model/' subdirectory by default.  The models are run using tf.MonitoredTrainingSession and save checkpoints in './Model/Checkpoints/' periodically throughout the training process; models are automatically restored whenever checkpoints exist in this folder.

The training and validation loss can be monitored using TensorBoard via:
```
$ tensorboard --logdir=./Model/logs
```
        
