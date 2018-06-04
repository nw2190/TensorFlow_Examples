import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

# Import MNIST loader and utility functions from 'utils.py' file
from utils import write_mnist_tfrecords, checkFolder, checkFolders, show_all_variables

# Import layer definitions from 'layers.py' file
from layers import dense, conv2d, conv2d_transpose, batch_norm

# Import Loader class and EarlyStoppingHook from 'misc.py' file
from misc import Loader, EarlyStoppingHook

# Import Flags specifying model hyperparameters and training options
from flags import getFlags_VAE


# Class representation of network model
class Model(object):
    
    # Initialize model
    def __init__(self, data_count, flags):
        self.data_count = data_count
        
        # Read keys/values from 'flags' and assign to self
        for key, val in flags.__dict__.items():
            if key not in self.__dict__.keys():
                self.__dict__[key] = val
                                        
        # Initialize loader and retrieve training, validation, and early stopping datasets
        self.loader = Loader(batch_size=self.batch_size)
        self.dataset, self.vdataset, self.edataset = self.loader.get_datasets()

        # Define tensor for updating global step
        self.global_step = tf.train.get_or_create_global_step()

        # Build graph for network model
        self.build_model()


    # Specify session for model evaluations
    def set_session(self, sess):
        self.sess = sess

    # Reinitialize handles for datasets when restoring from checkpoint
    def reinitialize_handles(self):
        self.training_handle = self.sess.run(self.dataset.string_handle())
        self.validation_handle = self.sess.run(self.vdataset.string_handle())

    # Encoder component of VAE model
    def encoder(self, x, training=True, reuse=None, name=None):

        # [None, 28, 28, 1]  -->  [None, 14, 14, 64]
        h = conv2d(x, 64, kernel_size=4, strides=2, activation=tf.nn.leaky_relu, reuse=reuse, name='e_conv_1')

        # [None, 14, 14, 64] -->  [None, 7, 7, 128]
        h = conv2d(h, 128, kernel_size=4, strides=2, reuse=reuse, name='e_conv_2')
        h = batch_norm(h, training=training, reuse=reuse, name='e_bn_1')
        h = tf.nn.leaky_relu(h)

        # [None, 7, 7, 128]  -->  [None, 7*7*128]
        h = tf.reshape(h, [-1, 7*7*128])

        # [None, 7*7*128] -->  [None, 1024]
        h = dense(h, 1024, reuse=reuse, name='e_dense_1')
        h = batch_norm(h, training=training, reuse=reuse, name='e_bn_2')
        h = tf.nn.leaky_relu(h)

        # [None, 1024] -->  [None, 2*self.z_dim]
        h = dense(h, 2*self.z_dim, reuse=reuse, name='e_dense_2')

        # Assign names to final outputs
        mean = tf.identity(h[:,:self.z_dim], name=name+"_mean")
        log_sigma = tf.identity(h[:,self.z_dim:], name=name+"_log_sigma")
        return mean, log_sigma

    # Decoder component of VAE model
    def decoder(self, z, training=True, reuse=None, name=None):

        # [None, z_dim]  -->  [None, 1024]
        h = dense(z, 1024, reuse=reuse, name='d_dense_1')
        h = batch_norm(h, training=training, reuse=reuse, name='d_bn_1')
        h = tf.nn.relu(h)
        
        # [None, 1024]  -->  [None, 7*7*128]
        h = dense(h, self.min_res*self.min_res*self.min_chans, reuse=reuse, name='d_dense_2')
        h = batch_norm(h, training=training, reuse=reuse, name='d_bn_2')
        h = tf.nn.relu(h)

        # [None, 7*7*128]  -->  [None, 7, 7, 128]
        h = tf.reshape(h, [-1, self.min_res, self.min_res, self.min_chans])

        # [None, 7, 7, 128]  -->  [None, 14, 14, 64]
        h = conv2d_transpose(h, 64, kernel_size=4, strides=2, reuse=reuse, name='d_tconv_1')
        h = batch_norm(h, training=training, reuse=reuse, name='d_bn_3')
        h = tf.nn.relu(h)
                        
        # [None, 14, 14, 64]  -->  [None, 28, 28, 1]
        h = conv2d_transpose(h, 1, kernel_size=4, strides=2, activation=tf.nn.sigmoid, reuse=reuse, name='d_tconv_2')
                        
        # Assign name to final output
        return tf.identity(h, name=name)

    # Sample from multivariate Gaussian
    def sampleGaussian(self, mean, log_sigma, name=None):
        epsilon = tf.random_normal(tf.shape(log_sigma))
        return tf.identity(mean + epsilon * tf.exp(log_sigma), name=name)

    # Define sampler for generating self.z values
    def sample_z(self, batch_size):
        return np.random.normal(size=(batch_size, self.z_dim))

    # Compute marginal likelihood loss
    def compute_ml_loss(self, data, pred):
        ml_loss = -tf.reduce_mean(tf.reduce_sum(data*tf.log(pred) + \
                                                (1 - data)*tf.log(1 - pred), [1, 2, 3]))
        return ml_loss

    # Compute Kullback–Leibler (KL) divergence
    def compute_kl_loss(self, mean, log_sigma):
        kl_loss = 0.5*tf.reduce_mean(tf.reduce_sum(tf.square(mean) + \
                                                   tf.square(tf.exp(log_sigma)) - \
                                                   2.*log_sigma - 1., axis=[-1]))
        return kl_loss
    
    # Define graph for model
    def build_model(self):
        """
        Network model adapted from VAE.py file in GitHub repo by 'hwalsuklee':
        https://github.com/hwalsuklee/tensorflow-generative-model-collections/blob/master/VAE.py
        """
        # Define placeholder for noise vector
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        
        # Define placeholder for dataset handle (e.g. 'training', 'validation', 'early_stopping')
        self.dataset_handle = tf.placeholder(tf.string, shape=[], name='dataset_handle')
        self.iterator = tf.data.Iterator.from_string_handle(self.dataset_handle, self.dataset.output_types, self.dataset.output_shapes)
        self.data = self.iterator.get_next()
        
        # Define tensor for retrieving early stopping dataset handle (for EarlyStoppingHook)
        self.stopping_string_handle = tf.identity(self.edataset.string_handle(), name='stopping_string_handle')

        # Define placeholder for learning rate and training status
        self.learning_rt = tf.placeholder(tf.float32, name='learning_rt')
        self.training = tf.placeholder(tf.bool, name='training')

        # Encode input images
        self.mean, self.log_sigma = self.encoder(self.data, training=self.training, reuse=None, name="encoder")
        self.emean, self.elog_sigma = self.encoder(self.edataset.get_next(), training=False, reuse=True, name="stopping_encoder")

        # Sample latent vector
        self.z_sample = self.sampleGaussian(self.mean, self.log_sigma, name="latent_vector")
        self.ez_sample = self.sampleGaussian(self.emean, self.elog_sigma, name="stopping_latent_vector")

        # Decode latent vector back to original image
        self.pred = self.decoder(self.z_sample, training=self.training, reuse=None, name="decoder")
        self.pred_sample = self.decoder(self.z, training=self.training, reuse=True, name="sampling_decoder")
        self.epred = self.decoder(self.ez_sample, training=False, reuse=True, name="stopping_decoder")        

        # Compute marginal likelihood loss
        self.ml_loss = self.compute_ml_loss(self.data, self.pred)
        self.eml_loss = self.compute_ml_loss(self.edataset.get_next(), self.epred)
        
        # Compute Kullback–Leibler (KL) divergence
        self.kl_loss = self.compute_kl_loss(self.mean, self.log_sigma)
        self.ekl_loss = self.compute_kl_loss(self.emean, self.elog_sigma)
                
        # Define loss according to the evidence lower bound objective (ELBO)
        self.loss = tf.add(self.ml_loss, self.kl_loss, name="loss")
        self.eloss = tf.add(self.eml_loss, self.ekl_loss, name="stopping_loss")
        
        # Define optimizers for training the discriminator and generator
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.adam_beta1) \
                                 .minimize(self.loss, global_step=self.global_step)
        
        # Define summary operations
        loss_sum = tf.summary.scalar("loss", self.loss)
        kl_loss_sum = tf.summary.scalar("kl_loss", self.kl_loss)
        ml_loss_sum = tf.summary.scalar("ml_loss", self.ml_loss)
        self.merged_summaries = tf.summary.merge([loss_sum, kl_loss_sum, ml_loss_sum])

        # Resize predictions for plotting
        self.resized_imgs = tf.image.resize_images(self.pred_sample, [self.plot_res, self.plot_res])

        
    # Train model
    def train(self):

        # Define summary writer for saving log files (for training and validation)
        self.writer = tf.summary.FileWriter(self.log_dir + 'training/', graph=tf.get_default_graph())
        self.vwriter = tf.summary.FileWriter(self.log_dir + 'validation/', graph=tf.get_default_graph())

        # Show list of all variables and total parameter count
        show_all_variables()
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
                self.learning_rate = self.lr_decay_rate*self.learning_rate

            # Specify feed dictionary
            fd = {self.dataset_handle: self.training_handle, self.z: np.zeros([self.batch_size, self.z_dim]),
                  self.learning_rt: self.learning_rate, self.training: True}

            # Save summariesm display progress and update model
            if (step % self.summary_step == 0) and (step % self.display_step == 0):
                summary, kl_loss, ml_loss, loss, _ = self.sess.run([self.merged_summaries, self.kl_loss, self.ml_loss,
                                                                    self.loss, self.optim], feed_dict=fd)
                print("Step %d:  %.10f [kl_loss]   %.10f [ml_loss]   %.10f [loss] " %(step,kl_loss,ml_loss,loss))
                self.writer.add_summary(summary, step); self.writer.flush()
            # Save summaries and update model
            elif step % self.summary_step == 0:
                summary, _ = self.sess.run([self.merged_summaries, self.optim], feed_dict=fd)
                self.writer.add_summary(summary, step); self.writer.flush()
            # Display progress and update model
            elif step % self.display_step == 0:
                kl_loss, ml_loss, loss, _ = self.sess.run([self.kl_loss, self.ml_loss,
                                                           self.loss, self.optim], feed_dict=fd)
                print("Step %d:  %.10f [kl_loss]   %.10f [ml_loss]   %.10f [loss] " %(step,kl_loss,ml_loss,loss))
            # Update model
            else:
                self.sess.run([self.optim], feed_dict=fd)

            # Plot predictions
            if step % self.plot_step == 0:
                self.plot_predictions(step)

            # Save validation summaries
            if step % self.summary_step == 0:
                fd = {self.dataset_handle: self.validation_handle, self.z: np.zeros([self.batch_size, self.z_dim]),
                      self.training: False}
                vsummary = self.sess.run(self.merged_summaries, feed_dict=fd)
                self.vwriter.add_summary(vsummary, step); self.vwriter.flush()
            
    # Define method for computing model predictions
    def predict(self):
        fd = {self.z: self.sample_z(self.batch_size), self.training: False}
        return self.sess.run(self.resized_imgs, feed_dict=fd)

    # Compute cumulative loss over multiple batches
    def cumulative_loss(self, loss, loss_ops, dataset_handle, batches):
        for n in range(0, batches):
            fd = {self.dataset_handle: dataset_handle, self.training: False}
            current_loss = self.sess.run(loss_ops, feed_dict=fd)
            loss = np.add(loss, current_loss)
        return loss
            
    # Evaluate model
    def evaluate(self):
        t_batches = int(np.floor(0.8 * self.data_count/self.batch_size))
        v_batches = int(np.floor(0.2 * self.data_count/self.batch_size))
        training_loss = self.cumulative_loss([0.], [self.loss], self.training_handle, t_batches)
        validation_loss = self.cumulative_loss([0.], [self.loss], self.validation_handle, v_batches)
        training_loss = training_loss/t_batches
        validation_loss = validation_loss/v_batches
        return training_loss, validation_loss

    # Plot generated images for qualitative evaluation
    def plot_predictions(self, step):
        plot_subdir = self.plot_dir + str(step) + "/"
        checkFolders([self.plot_dir, plot_subdir])
        resized_imgs = self.predict()
        for n in range(0, self.batch_size):
            plt.imsave(plot_subdir + 'plot_' + str(n) + '.png', resized_imgs[n,:,:,0], cmap='gray')
        
                    
# Initialize and train model 
def main():

    # Define model parameters and options in dictionary of flags
    FLAGS = getFlags_VAE()

    # Create tfrecords if file does not exist
    if not os.path.exists('./data/training.tfrecords'):
        print("\n [ Creating tfrecords files ]\n")
        write_mnist_tfrecords()
    
    # Initialize model
    model = Model(70000, FLAGS)

    # Specify number of training steps
    training_steps = FLAGS.__dict__['training_steps']

    # Define feed dictionary and loss name for EarlyStoppingHook
    loss_name = "stopping_loss:0"
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

        # Plot final predictions
        model.plot_predictions("final")

        # Reinitialize dataset handles
        model.reinitialize_handles()
        
        # Evaluate model
        print("Evaluating Model:")
        t_loss, v_loss = model.evaluate()
        print("Training loss: %.5f" %(t_loss))
        print("Validation loss: %.5f\n" %(v_loss))
        

# Run main() function when called directly
if __name__ == '__main__':
    main()
