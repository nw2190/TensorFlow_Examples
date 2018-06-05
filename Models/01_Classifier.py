import tensorflow as tf
import numpy as np
import sys
import os

# Import MNIST loader and utility functions from 'utils.py' file
from utils import write_mnist_tfrecords, checkFolders, show_variables, add_suffix

# Import layer definitions from 'layers.py' file
from layers import dense, conv2d, conv2d_transpose, batch_norm

# Import parse function for tfrecords features  and EarlyStoppingHook from 'misc.py' file
from misc import _parse_mnist_data, EarlyStoppingHook

# Import Flags specifying model hyperparameters and training options
from flags import getFlags_Classifier


# Class representation of network model
class Model(object):
    
    # Initialize model
    def __init__(self, data_count, flags):
        self.data_count = data_count

        # Read keys/values from flags and assign to self
        for key, val in flags.__dict__.items():
            if key not in self.__dict__.keys():
                self.__dict__[key] = val

        # Create tfrecords if file does not exist
        if not os.path.exists(os.path.join(self.data_dir,'training.tfrecords')):
            print("\n [ Creating tfrecords files ]\n")
            write_mnist_tfrecords(self.data_dir)

        # Initialize datasets for training, validation, and early stopping checks
        self.initialize_datasets()
        
        # Define tensor for updating global step
        self.global_step = tf.train.get_or_create_global_step()

        # Build graph for network model
        self.build_model()

    # Initialize datasets
    def initialize_datasets(self, stopping_size=14000):

        # Define iterator for training dataset
        self.dataset = tf.data.TFRecordDataset(os.path.join(self.data_dir, 'training.tfrecords'))
        self.dataset = self.dataset.map(_parse_mnist_data)
        self.dataset = self.dataset.apply(tf.contrib.data.shuffle_and_repeat(self.batch_size*5))
        self.dataset = self.dataset.batch(self.batch_size)
        self.dataset = self.dataset.prefetch(self.batch_size*5)
        self.dataset = self.dataset.make_one_shot_iterator()
        
        # Define iterator for training dataset
        self.vdataset = tf.data.TFRecordDataset(os.path.join(self.data_dir, 'validation.tfrecords'))
        self.vdataset = self.vdataset.map(_parse_mnist_data)
        self.vdataset = self.vdataset.apply(tf.contrib.data.shuffle_and_repeat(self.batch_size*5))
        self.vdataset = self.vdataset.batch(self.batch_size)
        self.vdataset = self.vdataset.prefetch(self.batch_size*5)
        self.vdataset = self.vdataset.make_one_shot_iterator()

        # Create early stopping batch from validation dataset
        self.edataset = tf.data.TFRecordDataset(os.path.join(self.data_dir, 'validation.tfrecords'))
        self.edataset = self.edataset.map(_parse_mnist_data)
        self.edataset = self.edataset.apply(tf.contrib.data.shuffle_and_repeat(stopping_size))
        self.edataset = self.edataset.batch(stopping_size)
        self.edataset = self.edataset.make_one_shot_iterator()

    # Specify session for model evaluations
    def set_session(self, sess):
        self.sess = sess

    # Reinitialize handles for datasets when restoring from checkpoint
    def reinitialize_handles(self):
        self.training_handle = self.sess.run(self.dataset.string_handle())
        self.validation_handle = self.sess.run(self.vdataset.string_handle())

    # Model classifier
    def classifier(self, x, training=True, reuse=None, name=None):

        # [None, 28, 28, 1]  -->  [None, 14, 14, 64]
        h = conv2d(x, 64, kernel_size=4, strides=2, activation=tf.nn.leaky_relu, reuse=reuse, name='c_conv_1')

        # [None, 14, 14, 64] -->  [None, 7, 7, 128]
        h = conv2d(h, 128, kernel_size=4, strides=2, reuse=reuse, name='c_conv_2')
        h = batch_norm(h, training=training, reuse=reuse, name='c_bn_1')
        h = tf.nn.leaky_relu(h)

        # [None, 7, 7, 128]  -->  [None, 7*7*128]
        h = tf.reshape(h, [-1, 7*7*128])

        # [None, 7*7*128] -->  [None, 1024]
        h = dense(h, 1024, reuse=reuse, name='c_dense_1')
        h = batch_norm(h, training=training, reuse=reuse, name='c_bn_2')
        h = tf.nn.leaky_relu(h)

        # [None, 1024] -->  [None, label_count]
        h = dense(h, self.label_count, reuse=reuse, name='c_dense_2')

        # Assign name to final output
        logits = tf.identity(h, name=name+"_logits")
        probs = tf.nn.sigmoid(logits, name=name+"_probs")
        return probs, logits

    # Compute sigmoid cross entropy loss
    def compute_cross_entropy(self, logits, labels, name=None):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels), name=name)

    # Evaluate model on specified batch of data
    def evaluate_model(self, data, reuse=None, training=True, suffix=None):

        # Unpack features from data
        images, labels = data
        
        # Classify input images
        probs, logits = self.classifier(images, training=training, reuse=reuse, name=add_suffix("classifier", suffix))

        # Compute sigmoid cross entropy loss
        loss = self.compute_cross_entropy(logits, labels, name=add_suffix("loss", suffix))
        
        return images, labels, probs, logits, loss
        
    # Define graph for model
    def build_model(self):

        # Define placeholder for dataset handle (to select training or validation)
        self.dataset_handle = tf.placeholder(tf.string, shape=[], name='dataset_handle')
        self.iterator = tf.data.Iterator.from_string_handle(self.dataset_handle, self.dataset.output_types, self.dataset.output_shapes)
        self.data = self.iterator.get_next()
        

        # Define placeholder for learning rate and training status
        self.learning_rt = tf.placeholder(tf.float32, name='learning_rt')
        self.training = tf.placeholder(tf.bool, name='training')

        # Compute predictions and loss for training/validation datasets
        self.images, self.labels, self.probs, self.logits, self.loss = self.evaluate_model(self.data, training=self.training)

        # Compute predictions and loss for early stopping checks
        _, __, self.eprobs, self.elogits, self.eloss = self.evaluate_model(self.edataset.get_next(), reuse=True,
                                                                           training=False, suffix="_stopping")

        # Define optimizer for training
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.adam_beta1) \
                                 .minimize(self.loss, global_step=self.global_step)
        
        # Define summary operations
        loss_sum = tf.summary.scalar("loss", self.loss)
        self.merged_summaries = tf.summary.merge_all()

        # Compute number of misclassified images for current batch
        true_labels = tf.argmax(self.labels, axis=-1)
        predictions = tf.argmax(self.probs, axis=-1)
        self.misclassifications = tf.reduce_sum(tf.cast(tf.logical_not(tf.equal(true_labels, predictions)), tf.int64))
        
    # Train model
    def train(self):

        # Define summary writer for saving log files (for training and validation)
        self.writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'training/'), graph=tf.get_default_graph())
        self.vwriter = tf.summary.FileWriter(os.path.join(self.log_dir, 'validation/'), graph=tf.get_default_graph())

        # Show list of all variables and total parameter count
        show_variables()
        print("\n[ Initializing Variables ]\n")

        # Get handles for training and validation datasets
        self.training_handle = self.sess.run(self.dataset.string_handle())
        self.validation_handle = self.sess.run(self.vdataset.string_handle())

        # Iterate through training steps
        while not self.sess.should_stop():

            # Update global step            
            step = tf.train.global_step(self.sess, self.global_step)

            # Break if early stopping hook requests stop after sess.run()
            if self.sess.should_stop():
                break

            # Apply decay to learning rate
            if step % self.lr_decay_step == 0:
                self.learning_rate = np.power(self.lr_decay_rate, step/self.lr_decay_step)*self.learning_rate

            # Specify feed dictionary
            fd = {self.dataset_handle: self.training_handle, self.learning_rt: self.learning_rate, self.training: True}

            # Save summaries, display progress, and update model
            if (step % self.summary_step == 0) and (step % self.display_step == 0):
                summary, loss, _ = self.sess.run([self.merged_summaries, self.loss, self.optim], feed_dict=fd)
                print("Step %d:  %.10f [loss] " %(step,loss))
                self.writer.add_summary(summary, step); self.writer.flush()
            # Save summaries and update model
            elif step % self.summary_step == 0:
                summary, _ = self.sess.run([self.merged_summaries, self.optim], feed_dict=fd)
                self.writer.add_summary(summary, step); self.writer.flush()
            # Display progress and update model
            elif step % self.display_step == 0:
                loss, _ = self.sess.run([self.loss, self.optim], feed_dict=fd)
                print("Step %d:  %.10f [loss] " %(step,loss))
            # Update model
            else:
                self.sess.run([self.optim], feed_dict=fd)

            # Break if early stopping hook requests stop after sess.run()
            if self.sess.should_stop():
                break

            # Plot predictions
            if step % self.plot_step == 0:
                self.plot_predictions()

            # Save validation summaries
            if step % self.summary_step == 0:
                fd = {self.dataset_handle: self.validation_handle, self.training: False}
                vsummary = self.sess.run(self.merged_summaries, feed_dict=fd)
                self.vwriter.add_summary(vsummary, step); self.vwriter.flush()
            
    # Define method for computing model predictions
    def predict(self):
        fd = {self.dataset_handle: self.validation_handle, self.training: False}
        data, probs =  self.sess.run([self.data, self.probs], feed_dict=fd)
        _, labels = data
        labels = np.argmax(labels, axis=-1)
        predictions = np.argmax(probs, axis=-1)
        return labels, predictions

    # Plot true labels and network predictions
    def plot_predictions(self):
        labels, predictions = self.predict()
        print('\n{0:^7}'.format('Label') + '|' + '{0:^7}'.format('Pred'))
        print('-'*15)
        for n in range(0, self.plot_count):
            print('{0:^7}'.format(labels[n]) + '|' + '{0:^7}'.format(predictions[n]))
        print('\n')
        
    # Compute cumulative loss over multiple batches
    def compute_cumulative_loss(self, loss, loss_ops, dataset_handle, batches):
        for n in range(0, batches):
            fd = {self.dataset_handle: dataset_handle, self.training: False}
            current_loss = self.sess.run(loss_ops, feed_dict=fd)
            loss = np.add(loss, current_loss)
            sys.stdout.write('Batch {0} of {1}\r'.format(n+1,batches))
            sys.stdout.flush()
        return loss
            
    # Evaluate model
    def evaluate(self):
        t_batches = int(np.floor(0.8 * self.data_count/self.batch_size))
        v_batches = int(np.floor(0.2 * self.data_count/self.batch_size))
        print("\nTraining dataset:")
        t_loss, t_misclass = self.compute_cumulative_loss([0.,0.], [self.loss,
                                                                    self.misclassifications], self.training_handle, t_batches)
        print("\n\nValidation dataset:")
        v_loss, v_misclass = self.compute_cumulative_loss([0.,0.], [self.loss,
                                                                    self.misclassifications], self.validation_handle, v_batches)
        training_loss = t_loss/t_batches
        validation_loss = v_loss/v_batches
        return training_loss, validation_loss, t_misclass, v_misclass, t_batches, v_batches, self.batch_size
        
                    
# Initialize and train model 
def main():

    # Define model parameters and options in dictionary of flags
    FLAGS = getFlags_Classifier()
    
    # Initialize model
    model = Model(70000, FLAGS)

    # Specify number of training steps
    training_steps = FLAGS.__dict__['training_steps']

    # Define feed dictionary and loss name for EarlyStoppingHook
    loss_name = "loss_stopping:0"
    start_step = FLAGS.__dict__['early_stopping_start']
    stopping_step = FLAGS.__dict__['early_stopping_step']
    tolerance = FLAGS.__dict__['early_stopping_tol']
    
    # Define saver which only keeps previous 3 checkpoints (default=10)
    scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=3))
    
    # Initialize TensorFlow monitored training session
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir = FLAGS.__dict__['checkpoint_dir'],
            hooks = [tf.train.StopAtStepHook(last_step=training_steps),
                     EarlyStoppingHook(loss_name, tolerance=tolerance, stopping_step=stopping_step, start_step=start_step)],
            save_summaries_steps = None, save_summaries_secs = None, save_checkpoint_secs = None,
            save_checkpoint_steps = FLAGS.__dict__['checkpoint_step'], scaffold=scaffold) as sess:

        # Set model session
        model.set_session(sess)
        
        # Train model
        model.train()

    print("\n[ TRAINING COMPLETE ]\n")

    # Create new session for model evaluation
    with tf.Session() as sess:

        # Restore network parameters from latest checkpoint
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.__dict__['checkpoint_dir']))
            
        # Set model session using restored sess
        model.set_session(sess)

        # Reinitialize dataset handles
        model.reinitialize_handles()

        # Evaluate model
        print("[ Evaluating Model ]")
        t_loss, v_loss, t_mis, v_mis, t_batch, v_batch, bs = model.evaluate()

        print("\n\n[ Final Evaluations ]")
        print("Training loss: %.5f  [ %d / %d misclassified ]" %(t_loss, t_mis, t_batch*bs))
        print("Validation loss: %.5f  [ %d / %d misclassified ]\n" %(v_loss, v_mis, v_batch*bs))
        

# Run main() function when called directly
if __name__ == '__main__':
    main()
