import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
import os

# Import MNIST loader and utility functions from 'utils.py' file
from utils import write_mnist_tfrecords, checkFolders, checkData, show_variables, add_suffix

# Import layer definitions from 'layers.py' file
from layers import dense, conv2d, conv2d_transpose, batch_norm

# Import parse function for tfrecords features from 'misc.py' file
from misc import _parse_mnist_image

# Import Flags specifying model hyperparameters and training options
from flags import getFlags_GAN


# Class representation of network model
class Model(object):
    
    # Initialize model
    def __init__(self, data_count, flags):
        self.data_count = data_count
        
        # Read keys/values from flags and assign to self
        for key, val in flags.__dict__.items():
            if key not in self.__dict__.keys():
                self.__dict__[key] = val
                                        
        # Check that data folder exists
        checkData(self.data_dir)

        # Create tfrecords if file does not exist
        if not os.path.exists(os.path.join(self.data_dir,'training.tfrecords')):
            print("\n [ Creating tfrecords files ]\n")
            write_mnist_tfrecords(self.data_dir)

        # Initialize training and validation datasets
        self.initialize_datasets()

        # Define tensor for updating global step
        self.global_step = tf.train.get_or_create_global_step()

        # Build graph for network model
        self.build_model()

    # Initialize datasets
    def initialize_datasets(self, stopping_size=14000):

        # Define iterator for training dataset
        self.dataset = tf.data.TFRecordDataset('./data/training.tfrecords')
        self.dataset = self.dataset.map(_parse_mnist_image)
        self.dataset = self.dataset.apply(tf.contrib.data.shuffle_and_repeat(self.batch_size*5))
        self.dataset = self.dataset.batch(self.batch_size)
        self.dataset = self.dataset.prefetch(self.batch_size*5)
        self.dataset = self.dataset.make_one_shot_iterator()
        
        # Define iterator for training dataset
        self.vdataset = tf.data.TFRecordDataset('./data/validation.tfrecords')
        self.vdataset = self.vdataset.map(_parse_mnist_image)
        self.vdataset = self.vdataset.apply(tf.contrib.data.shuffle_and_repeat(self.batch_size*5))
        self.vdataset = self.vdataset.batch(self.batch_size)
        self.vdataset = self.vdataset.prefetch(self.batch_size*5)
        self.vdataset = self.vdataset.make_one_shot_iterator()

    # Specify session for model evaluations
    def set_session(self, sess):
        self.sess = sess

    # Reinitialize handles for datasets when restoring from checkpoint
    def reinitialize_handles(self):
        self.training_handle = self.sess.run(self.dataset.string_handle())
        self.validation_handle = self.sess.run(self.vdataset.string_handle())

    # Generator component of GAN model
    def generator(self, z, training=True, reuse=None, name=None):

        # [None, z_dim]  -->  [None, 1024]
        h = dense(z, 1024, reuse=reuse, name='g_dense_1')
        h = batch_norm(h, training=training, reuse=reuse, name='g_bn_1')
        h = tf.nn.relu(h)
        
        # [None, 1024]  -->  [None, 7*7*128]
        h = dense(h, self.g_res*self.g_res*self.g_chans, reuse=reuse, name='g_dense_2')
        h = batch_norm(h, training=training, reuse=reuse, name='g_bn_2')
        h = tf.nn.relu(h)

        # [None, 7*7*128]  -->  [None, 7, 7, 128]
        h = tf.reshape(h, [-1, self.g_res, self.g_res, self.g_chans])

        # [None, 7, 7, 128]  -->  [None, 14, 14, 64]
        h = conv2d_transpose(h, 64, kernel_size=4, strides=2, reuse=reuse, name='g_tconv_1')
        h = batch_norm(h, training=training, reuse=reuse, name='g_bn_3')
        h = tf.nn.relu(h)
                        
        # [None, 14, 14, 64]  -->  [None, 28, 28, 1]
        h = conv2d_transpose(h, 1, kernel_size=4, strides=2, activation=tf.nn.sigmoid, reuse=reuse, name='g_tconv_2')
                        
        # Assign name to final output
        return tf.identity(h, name=name)

    # Discriminator component of GAN model
    def discriminator(self, x, training=True, reuse=None, name=None):

        # [None, 28, 28, 1]  -->  [None, 14, 14, 64]
        h = conv2d(x, 64, kernel_size=4, strides=2, activation=tf.nn.leaky_relu, reuse=reuse, name='d_conv_1')

        # [None, 14, 14, 64] -->  [None, 7, 7, 128]
        h = conv2d(h, 128, kernel_size=4, strides=2, reuse=reuse, name='d_conv_2')
        h = batch_norm(h, training=training, reuse=reuse, name='d_bn_1')
        h = tf.nn.leaky_relu(h)

        # [None, 7, 7, 128]  -->  [None, 7*7*128]
        h = tf.reshape(h, [-1, 7*7*128])

        # [None, 7*7*128] -->  [None, 1024]
        h = dense(h, 1024, reuse=reuse, name='d_dense_1')
        h = batch_norm(h, training=training, reuse=reuse, name='d_bn_2')
        h = tf.nn.leaky_relu(h)

        # [None, 1024] -->  [None, 1]
        h = dense(h, 1, reuse=reuse, name='d_dense_2')

        # Assign name to final output
        logits = tf.identity(h, name=name+"_logits")
        probs = tf.nn.sigmoid(logits, name=name+"_probs")
        return probs, logits

    # Define sampler for generating self.z values
    def sample_z(self, batch_size):
        return np.random.uniform(-1., 1., size=(batch_size, self.z_dim))

    # Compute sigmoid cross entropy loss
    def compute_cross_entropy(self, logits, labels, name=None):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels), name=name)
    
    # Evaluate model on specified batch of data
    def evaluate_model(self, z, data, reuse=None, training=True, suffix=None):
        
        # Generate predicted images from noisy latent vector z
        pred = self.generator(z, training=training, reuse=reuse, name=add_suffix("generator", suffix))

        # Compute discriminator probabilities/logits for fake images
        D_fake, D_fake_logits = self.discriminator(pred, training=training, reuse=reuse, name=add_suffix("D_fake", suffix))
        
        # Compute generator loss
        g_loss = self.compute_cross_entropy(D_fake_logits, tf.ones_like(D_fake), name=add_suffix("g_loss", suffix))

        # Compute discriminator loss for identifying fake images
        d_loss_fake = self.compute_cross_entropy(D_fake_logits, tf.zeros_like(D_fake))

        # Compute discriminator probabilities/logits for real images
        D_real, D_real_logits = self.discriminator(data, training=training, reuse=True, name=add_suffix("D_real", suffix))

        # Compute discriminator loss for identifying real images
        d_loss_real = self.compute_cross_entropy(D_real_logits, tf.ones_like(D_real))

        # Compute discriminator loss
        d_loss = tf.add(d_loss_real, d_loss_fake, name=add_suffix("d_loss", suffix))

        return pred, d_loss, g_loss

    # Define graph for model
    def build_model(self):
        """
        Network model adapted from GAN.py file in GitHub repo by 'hwalsuklee':
        https://github.com/hwalsuklee/tensorflow-generative-model-collections/blob/master/GAN.py
        """
        # Define placeholders for input and ouput values
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        # Define placeholder for dataset handle (to select training or validation)
        self.dataset_handle = tf.placeholder(tf.string, shape=[], name='dataset_handle')
        self.iterator = tf.data.Iterator.from_string_handle(self.dataset_handle, self.dataset.output_types, self.dataset.output_shapes)
        self.data = self.iterator.get_next()

        # Define placeholder for learning rate and training status
        self.learning_rt = tf.placeholder(tf.float32, name='learning_rt')
        self.training = tf.placeholder(tf.bool, name='training')

        # Compute predictions and loss for training/validation datasets
        self.pred, self.d_loss, self.g_loss = self.evaluate_model(self.z, self.data, training=self.training)
        
        # Create separate lists for discriminator and generator variables
        d_vars = [var for var in tf.trainable_variables() if 'd_' in var.name]
        g_vars = [var for var in tf.trainable_variables() if 'g_' in var.name]

        # Define optimizers for training the discriminator and generator
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.adam_beta1) \
                      .minimize(self.d_loss, var_list=d_vars, global_step=self.global_step)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.adam_beta1) \
                      .minimize(self.g_loss, var_list=g_vars)

        # Define summary operations
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.merged_summaries = tf.summary.merge([d_loss_sum, g_loss_sum])


        # Compute predictions from random samples in latent space and resize for plotting
        self.pred_sample = self.generator(self.z, training=False, reuse=True, name="sampling_generator")
        self.resized_pred = tf.image.resize_images(self.pred_sample, [self.plot_res, self.plot_res])

        
    # Train model
    def train(self):

        # Define summary writer for saving log files (for training and validation)
        self.writer = tf.summary.FileWriter(self.log_dir + 'training/', graph=tf.get_default_graph())
        self.vwriter = tf.summary.FileWriter(self.log_dir + 'validation/', graph=tf.get_default_graph())

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
            
            # Apply decay to learning rate
            if step % self.lr_decay_step == 0:
                self.learning_rate = self.lr_decay_rate*self.learning_rate

            # Generate random samples for generator input and specify feed dictionary
            z_batch = self.sample_z(self.batch_size)
            fd = {self.dataset_handle: self.training_handle, self.z: z_batch,
                  self.learning_rt: self.learning_rate, self.training: True}

            # Save summariesm display progress and update model
            if (step % self.summary_step == 0) and (step % self.display_step == 0):
                summary, d_loss, g_loss, _, __ = self.sess.run([self.merged_summaries, self.d_loss, self.g_loss,
                                                                self.d_optim, self.g_optim], feed_dict=fd)
                print("Step %d:  %.10f [d_loss]   %.10f [g_loss] " %(step,d_loss,g_loss))
                self.writer.add_summary(summary, step); self.writer.flush()
            # Save summaries and update model
            elif step % self.summary_step == 0:
                summary, _, __ = self.sess.run([self.merged_summaries, self.d_optim, self.g_optim], feed_dict=fd)
                self.writer.add_summary(summary, step); self.writer.flush()
            # Display progress and update model
            elif step % self.display_step == 0:
                d_loss, g_loss, _, __ = self.sess.run([self.d_loss, self.g_loss,
                                                       self.d_optim, self.g_optim], feed_dict=fd)
                print("Step %d:  %.10f [d_loss]   %.10f [g_loss] " %(step,d_loss,g_loss))
            # Update model
            else:
                self.sess.run([self.d_optim, self.g_optim], feed_dict=fd)

            # Plot predictions
            if step % self.plot_step == 0:
                self.plot_predictions(step)

            # Save validation summaries
            if step % self.summary_step == 0:
                fd = {self.dataset_handle: self.validation_handle, self.z: z_batch, self.training: False}
                vsummary = self.sess.run(self.merged_summaries, feed_dict=fd)
                self.vwriter.add_summary(vsummary, step); self.vwriter.flush()
            
    # Define method for computing model predictions
    def predict(self):
        fd = {self.dataset_handle: self.validation_handle, self.z: self.sample_z(self.batch_size), self.training: False}
        return self.sess.run(self.resized_pred, feed_dict=fd)

    # Plot generated images for qualitative evaluation
    def plot_predictions(self, step):
        plot_subdir = self.plot_dir + str(step) + "/"
        checkFolders([self.plot_dir, plot_subdir])
        resized_imgs = self.predict()
        for n in range(0, self.batch_size):
            plt.imsave(plot_subdir + 'plot_' + str(n) + '.png', resized_imgs[n,:,:,0], cmap='gray')

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
        t_loss, g_loss = self.compute_cumulative_loss([0.,0.], [self.d_loss, self.g_loss], self.training_handle, t_batches)
        print("\n\nValidation dataset:")
        v_loss, g_loss = self.compute_cumulative_loss([0.,g_loss], [self.d_loss, self.g_loss], self.validation_handle, v_batches)
        training_loss = t_loss/t_batches
        validation_loss = v_loss/v_batches
        generator_loss = g_loss/(t_batches+v_batches)
        return training_loss, validation_loss, generator_loss
        
                    
# Initialize and train model 
def main():

    # Define model parameters and options in dictionary of flags
    FLAGS = getFlags_GAN()

    # Initialize model
    model = Model(70000, FLAGS)

    # Specify number of training steps
    training_steps = FLAGS.__dict__['training_steps']

    # Define saver which only keeps previous 3 checkpoints (default=10)
    scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=3))
    
    # Initialize TensorFlow monitored training session
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir = FLAGS.__dict__['checkpoint_dir'],
            hooks = [tf.train.StopAtStepHook(last_step=training_steps)],
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

        # Plot final predictions
        model.plot_predictions("final")

        # Reinitialize dataset handles
        model.reinitialize_handles()
        
        # Evaluate model
        print("[ Evaluating Model ]")
        t_loss, v_loss, g_loss = model.evaluate()

        print("\n\n[ Final Evaluations ]")
        print("Training loss: %.5f" %(t_loss))
        print("Validation loss: %.5f" %(v_loss))
        print("Generator loss: %.5f\n" %(g_loss))


# Run main() function when called directly
if __name__ == '__main__':
    main()
