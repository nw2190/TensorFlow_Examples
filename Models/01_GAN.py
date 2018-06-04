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
from flags import getFlags_GAN


# Class representation of network model
class Model(object):
    
    # Initialize model
    def __init__(self, data_count, flags):
        self.data_count = data_count
        
        # Read keys/values from 'flags' and assign to self
        for key, val in flags.__dict__.items():
            if key not in self.__dict__.keys():
                self.__dict__[key] = val
                                        
        # Initialize loader and retrieve training and validation datasets
        self.loader = Loader(batch_size=self.batch_size)
        self.dataset, self.vdataset = self.loader.get_datasets()

        # Define tensor for updating global step
        self.global_step = tf.train.get_or_create_global_step()

        # Build graph for network model
        self.build_model()


    # Specify session for model evaluations
    def set_session(self, sess):
        self.sess = sess

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
        
    # Define graph for model
    def build_model(self):
        """
        Network model adapted from GAN.py file in GitHub repo by 'hwalsuklee':
        https://github.com/hwalsuklee/tensorflow-generative-model-collections/blob/master/GAN.py
        """
        # Define placeholders for input and ouput values
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
 
        # Define placeholder for learning rate and training status
        self.learning_rt = tf.placeholder(tf.float32, name='learning_rt')
        self.training = tf.placeholder(tf.bool, name='training')

        # Compute discriminator probabilities/logits for real images
        D_real, D_real_logits = self.discriminator(self.dataset, training=self.training, reuse=None, name="D_real")
        vD_real, vD_real_logits = self.discriminator(self.vdataset, training=self.training, reuse=True, name="vD_real")

        # Compute discriminator loss for identifying real images
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
        vd_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=vD_real_logits, labels=tf.ones_like(vD_real)))

        # Generate fake images from noisy latent vector self.z
        self.pred = self.generator(self.z, training=self.training, reuse=None, name="pred")

        # Compute discriminator probabilities/logits for fake images
        D_fake, D_fake_logits = self.discriminator(self.pred, training=self.training, reuse=True, name="D_fake")

        # Compute discriminator loss for identifying fake images
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))

        # Compute discriminator loss
        self.d_loss = tf.add(d_loss_real, d_loss_fake, name="d_loss")
        self.vd_loss = tf.add(vd_loss_real, d_loss_fake, name="vd_loss")

        # Compute generator loss
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)), name="g_loss")
        
        # Create separate lists for discriminator and generator variables
        d_vars = [var for var in tf.trainable_variables() if 'd_' in var.name]
        g_vars = [var for var in tf.trainable_variables() if 'g_' in var.name]

        # Define optimizers for training the discriminator and generator
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.adam_beta1) \
                      .minimize(self.d_loss, var_list=d_vars, global_step=self.global_step)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.adam_beta1) \
                      .minimize(self.g_loss, var_list=g_vars)#, global_step=self.global_step)

        # Define summary operations
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        vd_loss_sum = tf.summary.scalar("vd_loss", self.vd_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.merged_summaries = tf.summary.merge([d_loss_sum, g_loss_sum])
        self.vsummaries = tf.summary.merge([vd_loss_sum, g_loss_sum])

        # Resize predictions for plotting
        self.resized_imgs = tf.image.resize_images(self.pred, [self.plot_res, self.plot_res])

        
    # Train model
    def train(self):

        # Define summary writer for saving log files (for training and validation)
        self.writer = tf.summary.FileWriter(self.log_dir + 'training/', graph=tf.get_default_graph())
        self.vwriter = tf.summary.FileWriter(self.log_dir + 'validation/', graph=tf.get_default_graph())

        # Show list of all variables and total parameter count
        show_all_variables()
        print("\n[ Initializing Variables ]\n")
                    
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

            # Generate random samples for generator input and specify feed dictionary
            z_batch = self.sample_z(self.batch_size)
            fd = {self.z: z_batch, self.learning_rt: self.learning_rate, self.training: True}

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
                fd = {self.z: z_batch, self.training: False}
                vsummary = self.sess.run(self.vsummaries, feed_dict=fd)
                self.vwriter.add_summary(vsummary, step); self.vwriter.flush()
            
    # Define method for computing model predictions
    def predict(self):
        fd = {self.z: self.sample_z(self.batch_size), self.training: False}
        return self.sess.run(self.resized_imgs, feed_dict=fd)

    # Compute cumulative loss over multiple batches
    def cumulative_loss(self, loss, loss_ops, batches):
        for n in range(0, batches):
            fd = {self.z: self.sample_z(self.batch_size), self.training: False}
            current_loss = self.sess.run(loss_ops, feed_dict=fd)
            loss = np.add(loss, current_loss)
        return loss
            
    # Evaluate model
    def evaluate(self):
        t_batches = int(np.floor(0.8 * self.data_count/self.batch_size))
        v_batches = int(np.floor(0.2 * self.data_count/self.batch_size))
        g_loss, training_loss = self.cumulative_loss([0.,0.], [self.g_loss, self.d_loss], t_batches)
        g_loss, validation_loss = self.cumulative_loss([g_loss,0.], [self.g_loss, self.vd_loss], v_batches)
        training_loss = training_loss/t_batches
        validation_loss = validation_loss/v_batches
        g_loss = g_loss/(t_batches+v_batches)
        return training_loss, validation_loss, g_loss

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
    FLAGS = getFlags_GAN()

    # Create tfrecords if file does not exist
    if not os.path.exists('./data/training.tfrecords'):
        print("\n [ Creating tfrecords files ]\n")
        write_mnist_tfrecords()
    
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

        # Evaluate model
        print("EVALUATING MODEL:")
        t_loss, v_loss, g_loss = model.evaluate()
        print("Training loss: %.5f" %(t_loss))
        print("Validation loss: %.5f" %(v_loss))
        print("Generator loss: %.5f" %(g_loss))
        

# Run main() function when called directly
if __name__ == '__main__':
    main()
