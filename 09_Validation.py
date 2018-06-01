import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from random import shuffle

# Import base model for defining early stopping hook
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/session_run_hook.py
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util

# Class representation of network model
class Model(object):
    
    # Initialize model
    def __init__(self, x_data, y_data, batch_size):
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size

        # Initialize data loader
        self.dataset, self.vdataset = self.initialize_loaders()

        # Define tensor for updating global step
        self.global_step = tf.train.get_or_create_global_step()

        # Build graph for network model
        self.build_model()

    # Specify session for model evaluations
    def set_session(self, sess):
        self.sess = sess
        
    # Define loader for training dataset with mini-batch size 100
    def initialize_loaders(self):
        # Compute indices for training and validation
        indices = [n for n in range(0,self.x_data.shape[0])]
        shuffle(indices)
        t_indices = indices[0 : int(np.floor(0.8 * self.x_data.shape[0]))]
        v_indices = indices[int(np.floor(0.8 * self.x_data.shape[0])) : ]

        # Define training dataset loader
        dataset = tf.data.Dataset.from_tensor_slices((self.x_data[t_indices],self.y_data[t_indices]))
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(self.batch_size*5))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.batch_size*5)
        dataset = dataset.make_one_shot_iterator()
        dataset = dataset.get_next()

        # Define validation dataset loader
        vdataset = tf.data.Dataset.from_tensor_slices((self.x_data[v_indices],self.y_data[v_indices]))
        vdataset = vdataset.apply(tf.contrib.data.shuffle_and_repeat(self.batch_size*5))
        vdataset = vdataset.batch(self.batch_size)
        vdataset = vdataset.prefetch(self.batch_size*5)
        vdataset = vdataset.make_one_shot_iterator()
        vdataset = vdataset.get_next()
        return [dataset, vdataset]

    # Define neural network for model
    def network(self, X, training=True, reuse=None, name=None):
        # Define regularizer for weights
        wt_reg = tf.contrib.layers.l2_regularizer(0.0000001)
        
        # Define fully-connected layer with 10 hidden units
        h = tf.layers.dense(X, 10, activation=tf.nn.sigmoid,\
                            kernel_regularizer=wt_reg, reuse=reuse, name='dense_1')
        
        # Define batch normalization layer
        h = tf.layers.batch_normalization(h, scale=True, momentum=0.99,
                                          training=training, reuse=reuse, name='bn_1')
        
        # Define fully-connected layer with 20 hidden units
        h = tf.layers.dense(h, 20, activation=tf.nn.sigmoid,\
                            kernel_regularizer=wt_reg, reuse=reuse, name='dense_2')
        
        # Define batch normalization layer
        h = tf.layers.batch_normalization(h, scale=True, momentum=0.99,
                                          training=training, reuse=reuse, name='bn_2')

        # Define fully-connected layer with 10 hidden units
        h = tf.layers.dense(h, 10, activation=tf.nn.sigmoid,\
                            kernel_regularizer=wt_reg, reuse=reuse, name='dense_3')

        # Define fully-connected layer to single ouput prediction
        pred = tf.layers.dense(h, 1, activation=None,\
                               kernel_regularizer=wt_reg, reuse=reuse, name='dense_4')

        pred = tf.identity(pred, name=name)

        return pred

    # Define graph for model
    def build_model(self):

        # Define placeholders for input and ouput values
        self.x = tf.placeholder(tf.float32, [None, 1], name='x')
        self.y = tf.placeholder(tf.float32, [None, 1], name='y')

        # Define constants for early stopping dataset
        self.es_x = tf.constant(self.x_data, dtype=tf.float32, name='es_x')
        self.es_y = tf.constant(self.y_data, dtype=tf.float32, name='es_y')

        # Define placeholder for learning rate and training status
        self.learning_rt = tf.placeholder(tf.float32, name='learning_rt')
        self.training = tf.placeholder(tf.bool, name='training')

        # Define prediction to be output of network
        self.pred = self.network(self.x, training=self.training, name='pred')

        # Define mean square loss function
        self.ms_loss = tf.reduce_mean(tf.pow(self.pred - self.y, 2), name='ms_loss')
        
        # Define regularization loss
        self.reg_list = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_loss = tf.reduce_sum(self.reg_list, name='reg_loss')

        # Define total loss function
        self.loss = tf.add(self.ms_loss, self.reg_loss, name='loss')

        # Define tensors for early stopping
        self.es_pred = self.network(self.es_x, training=False, reuse=True)
        self.es_loss = tf.reduce_mean(tf.pow(self.es_pred - self.es_y, 2), name='es_loss')

        # Define update operations for batch normalization    
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        # Define optimizer
        with tf.control_dependencies(self.update_ops):
            self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rt).\
                minimize(self.loss, global_step=self.global_step)

        # Define summary operation for saving losses
        tf.summary.scalar("MS_Loss", self.ms_loss)
        tf.summary.scalar("REG_Loss", self.reg_loss)
        tf.summary.scalar("Loss", self.loss)
        self.merged_summaries = tf.summary.merge_all()
        
    # Train model
    def train(self):

        # Specify initial learning rate
        learning_rate = 0.0005

        # Define summary writer for saving log files (for training and validation)
        self.writer = tf.summary.FileWriter('./Model/logs/training/', graph=tf.get_default_graph())
        self.vwriter = tf.summary.FileWriter('./Model/logs/validation/', graph=tf.get_default_graph())

        # Iterate through 20000 training steps
        while not self.sess.should_stop():

            # Update global step
            step = tf.train.global_step(self.sess, self.global_step)

            # Retrieve batch from data loader
            x_batch, y_batch = self.sess.run(self.dataset)

            # Break if early stopping hook requests stop after sess.run()
            if self.sess.should_stop():
                break

            # Apply decay to learning rate every 1000 steps
            if step % 1000 == 0:
                learning_rate = 0.8*learning_rate

            # Run optimization operation for current mini-batch
            fd = {self.x: x_batch, self.y: y_batch,
                  self.learning_rt: learning_rate, self.training: True}
            self.sess.run(self.optim, feed_dict=fd)

            # Save summary every 100 steps
            if step % 100 == 0:
                summary = self.sess.run(self.merged_summaries, feed_dict=fd)
                self.writer.add_summary(summary, step)
                self.writer.flush()

            # Display progress every 1000 steps
            if step % 1000 == 0:
                loss = self.sess.run(self.loss, feed_dict=fd)
                print("Step %d:  %.10f" %(step,loss))

            # Compute validation loss every 100 steps
            if step % 100 == 0:
                x_vbatch, y_vbatch = self.sess.run(self.vdataset)
                fd = {self.x: x_vbatch, self.y: y_vbatch, self.training: False}
                vsummary = self.sess.run(self.merged_summaries, feed_dict=fd)
                self.vwriter.add_summary(vsummary, step)
                self.vwriter.flush()
                
    # Evaluate model
    def evaluate(self):
        
        # Compute final loss on full dataset
        fd = {self.x: self.x_data, self.y: self.y_data, self.training: False}
        final_loss = self.sess.run(self.ms_loss, feed_dict=fd)
        print("FINAL LOSS = %.10f" %(final_loss))

        # Plot predicted and true values for qualitative evaluation
        eval_pts = np.expand_dims(np.linspace(-np.pi/2, np.pi/2, 1000) , 1)
        predictions = self.sess.run(self.pred, feed_dict={self.x: eval_pts, self.training: False})
        true_values = np.sin(eval_pts)
        plt.plot(eval_pts[:,0], predictions[:,0], 'b')
        plt.plot(eval_pts[:,0], true_values[:,0], 'r')
        plt.scatter(self.x_data[0:10000,0], self.y_data[0:10000,0], alpha=0.05)
        plt.show()

        
# Define early stopping hook
class EarlyStoppingHook(session_run_hook.SessionRunHook):
    def __init__(self, tolerance=0.01):
        self.tolerance = tolerance

    # Initialize global and internal steps
    def begin(self):
        self._global_step_tensor = training_util._get_or_create_global_step_read()
        self._prev_step = -1
        self._step = 0

    # Evaluate early stopping loss every 1000 steps
    # (avoiding repetition when multiple run calls are made each step)
    def before_run(self, run_context):
        if (self._step % 1000 == 0) and (not self._step == self._prev_step):
            graph = run_context.session.graph
            loss_name = "es_loss:0"
            loss_tensor = graph.get_tensor_by_name(loss_name)
            return session_run_hook.SessionRunArgs({'step': self._global_step_tensor,
                                                    'loss': loss_tensor})
        else:
            return session_run_hook.SessionRunArgs({'step': self._global_step_tensor})
                                                    
    # Check if current loss is below tolerance for early stopping
    def after_run(self, run_context, run_values):
        if (self._step % 1000 == 0) and (not self._step == self._prev_step):
            global_step = run_values.results['step']
            current_loss = run_values.results['loss']
            if current_loss < self.tolerance:
                print("[Early Stopping Criterion Satisfied]")
                run_context.request_stop()
            self._prev_step = global_step
        else:
            global_step = run_values.results['step']
            self._step = global_step

            
# Initialize and train model 
def main():

    # Create artificial data
    x_data = np.pi/2 * np.random.normal(scale=0.333, size=[100*10000, 1])
    noise = np.random.normal(scale=0.05, size=[100*10000, 1])
    y_data = np.sin(x_data) + noise

    # Initialize model
    model = Model(x_data, y_data, 100)

    # Specify number of training steps
    training_steps = 20000

    # Initialize TensorFlow monitored training session
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir = "./Model/Checkpoints/",
            hooks = [tf.train.StopAtStepHook(last_step=training_steps),
                     EarlyStoppingHook(tolerance=0.0005)],
            save_summaries_steps = None,
            save_checkpoint_steps = 5000) as sess:

        # Set model session
        model.set_session(sess)

        # Train model
        model.train()


    # Create new session for model evaluation
    with tf.Session() as sess:

        # Restore network parameters from latest checkpoint
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint("./Model/Checkpoints/"))
            
        # Set model session using restored sess
        model.set_session(sess)

        # Evaluate model
        model.evaluate()

        
# Run main() function when called directly
if __name__ == '__main__':
    main()
