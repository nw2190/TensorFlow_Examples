import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

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
        self.dataset = self.initialize_loader()

        # Define tensor for updating global step
        self.global_step = tf.train.get_or_create_global_step()

        # Build graph for network model
        self.build_model()

    # Initialize session
    def set_session(self, sess):
        self.sess = sess
        
    # Define loader for training dataset with mini-batch size 100
    def initialize_loader(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.x_data,self.y_data))
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(self.batch_size*5))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.batch_size*5)
        dataset = dataset.make_one_shot_iterator()
        dataset = dataset.get_next()
        return dataset

    # Define neural network for model
    def network(self, X, reuse=None):
        # Define fully-connected layer with 10 hidden units
        h = tf.layers.dense(X, 10, activation=tf.nn.relu, reuse=reuse, name='dense_1')

        # Define fully-connected layer with 20 hidden units
        h = tf.layers.dense(h, 20, activation=tf.nn.relu, reuse=reuse, name='dense_2')

        # Define fully-connected layer with 10 hidden units
        h = tf.layers.dense(h, 10, activation=tf.nn.relu, reuse=reuse, name='dense_3')

        # Define fully-connected layer to single ouput prediction
        pred = tf.layers.dense(h, 1, activation=None, reuse=reuse, name='dense_4')

        return pred

    # Define graph for model
    def build_model(self):

        # Define placeholders for input and ouput values
        self.x = tf.placeholder(tf.float32, [None, 1], name='x')
        self.y = tf.placeholder(tf.float32, [None, 1], name='y')

        # Define constants for early stopping dataset
        self.es_x = tf.constant(self.x_data, dtype=tf.float32, name='es_x')
        self.es_y = tf.constant(self.y_data, dtype=tf.float32, name='es_y')

        # Define placeholder for learning rate
        self.learning_rt = tf.placeholder(tf.float32, name='learning_rt')

        # Define prediction to be output of network
        self.pred = self.network(self.x)

        # Define loss function
        self.loss = tf.reduce_mean(tf.pow(self.pred - self.y, 2), name='loss')

        # Define tensors for early stopping
        self.es_pred = self.network(self.es_x, reuse=True)
        self.es_loss = tf.reduce_mean(tf.pow(self.es_pred - self.es_y, 2), name='es_loss')

        # Define optimizer
        self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rt).\
            minimize(self.loss, global_step=self.global_step)

        # Define summary operation for saving losses
        tf.summary.scalar("Loss", self.loss)
        self.merged_summaries = tf.summary.merge_all()

        
    # Train model
    def train(self):

        # Specify initial learning rate
        learning_rate = 0.00075

        # Define summary writer for saving log files
        self.writer = tf.summary.FileWriter('./Model/logs/', graph=tf.get_default_graph())

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
                learning_rate = 0.9*learning_rate

            # Run optimization operation for current mini-batch
            fd = {self.x: x_batch, self.y: y_batch, self.learning_rt: learning_rate}
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

    # Define method for computing model predictions
    def predict(self, eval_pts):
        return self.sess.run(self.pred, feed_dict={self.x: eval_pts})
                
    # Evaluate model
    def evaluate(self):

        # Compute final loss on full dataset
        fd = {self.x: self.x_data, self.y: self.y_data}
        final_loss = self.sess.run(self.loss, feed_dict=fd)
        print("FINAL LOSS = %.10f" %(final_loss))

        # Plot predicted and true values for qualitative evaluation
        eval_pts = np.expand_dims(np.linspace(-np.pi/2, np.pi/2, 1000) , 1)
        predictions = self.predict(eval_pts)
        true_values = np.sin(eval_pts)
        plt.plot(eval_pts[:,0], predictions[:,0], 'b')
        plt.plot(eval_pts[:,0], true_values[:,0], 'r')
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
    y_data = np.sin(x_data)

    # Initialize model
    model = Model(x_data, y_data, 100)

    # Specify number of training steps
    training_steps = 20000

    # Initialize TensorFlow monitored training session
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir = "./Model/Checkpoints/",
            hooks = [tf.train.StopAtStepHook(last_step=training_steps),
                     EarlyStoppingHook(tolerance=0.000475)],
            save_summaries_steps = None,
            save_checkpoint_steps = 5000) as sess:

        # Initialize model session
        model.set_session(sess)

        # Train model
        model.train()


    # Create new session for model evaluation
    with tf.Session() as sess:

        # Restore network parameters from latest checkpoint
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint("./Model/Checkpoints/"))
            
        # Initialize model with new session
        model.set_session(sess)

        # Evaluate model
        model.evaluate()

        
# Run main() function when called directly
if __name__ == '__main__':
    main()
