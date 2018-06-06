import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from random import shuffle

# Class representation of network model
class Model(object):
    
    # Initialize model
    def __init__(self, x_data, y_data, learning_rate, batch_size):
        self.x_data = x_data
        self.y_data = y_data
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Initialize training and validation datasets
        self.initialize_datasets()

        # Define tensor for updating global step
        self.global_step = tf.train.get_or_create_global_step()

        # Build graph for network model
        self.build_model()

    # Specify session for model evaluations
    def set_session(self, sess):
        self.sess = sess
        
    # Initialize datasets
    def initialize_datasets(self):
        # Compute indices for training and validation
        indices = [n for n in range(0,self.x_data.shape[0])]
        shuffle(indices)
        t_indices = indices[0 : int(np.floor(0.8 * self.x_data.shape[0]))]
        v_indices = indices[int(np.floor(0.8 * self.x_data.shape[0])) : ]

        # Define training dataset loader
        self.dataset = tf.data.Dataset.from_tensor_slices((self.x_data[t_indices],self.y_data[t_indices]))
        self.dataset = self.dataset.apply(tf.contrib.data.shuffle_and_repeat(self.batch_size*5))
        self.dataset = self.dataset.batch(self.batch_size)
        self.dataset = self.dataset.prefetch(self.batch_size*5)
        self.dataset = self.dataset.make_one_shot_iterator()
        self.dataset = self.dataset.get_next()

        # Define validation dataset loader
        self.vdataset = tf.data.Dataset.from_tensor_slices((self.x_data[v_indices],self.y_data[v_indices]))
        self.vdataset = self.vdataset.apply(tf.contrib.data.shuffle_and_repeat(self.batch_size*5))
        self.vdataset = self.vdataset.batch(self.batch_size)
        self.vdataset = self.vdataset.prefetch(self.batch_size*5)
        self.vdataset = self.vdataset.make_one_shot_iterator()
        self.vdataset = self.vdataset.get_next()

        # Save training and validation indices
        self.t_indices = t_indices
        self.v_indices = v_indices

    # Define method for retrieving training dataset
    def get_train_data(self):
        return [self.x_data[self.t_indices], self.y_data[self.t_indices]]

    # Define method for retrieving validation dataset
    def get_val_data(self):
        return [self.x_data[self.v_indices], self.y_data[self.v_indices]]

    # Define neural network for model
    def network(self, X, training=True, reuse=None, name=None):
        # Define regularizer for weights
        wt_reg = tf.contrib.layers.l2_regularizer(0.0000001)
        
        # Define fully-connected layer with 10 hidden units
        h = tf.layers.dense(X, 10, activation=tf.nn.leaky_relu, kernel_regularizer=wt_reg, reuse=reuse, name='dense_1')
        
        # Define batch normalization layer followed by activation function
        h = tf.layers.batch_normalization(h, scale=True, momentum=0.9, training=training, reuse=reuse, name='bn_1')
        h = tf.nn.leaky_relu(h)
        
        # Define fully-connected layer with 20 hidden units
        h = tf.layers.dense(h, 20, kernel_regularizer=wt_reg, reuse=reuse, name='dense_2')
        
        # Define batch normalization layer followed by activation function
        h = tf.layers.batch_normalization(h, scale=True, momentum=0.9, training=training, reuse=reuse, name='bn_2')
        h = tf.nn.leaky_relu(h)
        
        # Define fully-connected layer with 10 hidden units and leaky_relu activation function
        h = tf.layers.dense(h, 10, kernel_regularizer=wt_reg, reuse=reuse, name='dense_3')

        # Define batch normalization layer followed by activation function
        h = tf.layers.batch_normalization(h, scale=True, momentum=0.9, training=training, reuse=reuse, name='bn_3')
        h = tf.nn.leaky_relu(h)

        # Define fully-connected layer to single ouput prediction
        pred = tf.layers.dense(h, 1, kernel_regularizer=wt_reg, reuse=reuse, name='dense_4')

        # Assign name to final output
        pred = tf.identity(pred, name=name)

        return pred

    # Define graph for model
    def build_model(self):

        # Define placeholders for input and ouput values
        self.x = tf.placeholder(tf.float32, [None, 1], name='x')
        self.y = tf.placeholder(tf.float32, [None, 1], name='y')

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

        # Define update operations for batch normalization    
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        # Define optimizer
        with tf.control_dependencies(self.update_ops):
            self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rt) \
                                 .minimize(self.loss, global_step=self.global_step)

        # Define summary operation for saving losses
        tf.summary.scalar("MS_Loss", self.ms_loss)
        tf.summary.scalar("REG_Loss", self.reg_loss)
        tf.summary.scalar("Loss", self.loss)
        self.merged_summaries = tf.summary.merge_all()
        
    # Train model
    def train(self):

        # Define summary writer for saving log files (for training and validation)
        self.writer = tf.summary.FileWriter('./Model/logs/training/', graph=tf.get_default_graph())
        self.vwriter = tf.summary.FileWriter('./Model/logs/validation/', graph=tf.get_default_graph())

        # Iterate through 20000 training steps
        while not self.sess.should_stop():

            # Update global step
            step = tf.train.global_step(self.sess, self.global_step)

            # Retrieve batch from data loader
            x_batch, y_batch = self.sess.run(self.dataset)

            # Apply decay to learning rate every 1000 steps
            if step % 1000 == 0:
                self.learning_rate = 0.9*self.learning_rate

            # Run optimization operation for current mini-batch
            fd = {self.x: x_batch, self.y: y_batch,
                  self.learning_rt: self.learning_rate, self.training: True}
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

    # Define method for computing model predictions
    def predict(self, eval_pts):
        return self.sess.run(self.pred, feed_dict={self.x: eval_pts, self.training: False})
                
    # Evaluate model
    def evaluate(self, x_data, y_data):
        fd = {self.x: x_data, self.y: y_data, self.training: False}
        current_loss = self.sess.run(self.ms_loss, feed_dict=fd)
        return current_loss

    # Plot predicted and true values for qualitative evaluation
    def plot_predictions(self):
        eval_pts = np.expand_dims(np.linspace(-np.pi/2, np.pi/2, 1000) , 1)
        predictions = self.predict(eval_pts)
        true_values = np.sin(eval_pts)
        plt.plot(eval_pts[:,0], predictions[:,0], 'b')
        plt.plot(eval_pts[:,0], true_values[:,0], 'r')
        plt.scatter(self.x_data[0:10000,0], self.y_data[0:10000,0], alpha=0.05)
        plt.show()

            
# Initialize and train model 
def main():

    # Create artificial data (no noise by default)
    x_data = np.pi/2 * np.random.normal(size=[100*10000, 1])
    #x_data = np.pi/2 * np.random.normal(scale=0.333, size=[100*10000, 1])
    noise = np.random.normal(scale=0.0, size=[100*10000, 1])
    y_data = np.sin(x_data) + noise

    # Specify initial learning rate
    learning_rate = 0.00075

    # Specify training batch size
    batch_size = 100

    # Initialize model
    model = Model(x_data, y_data, learning_rate, batch_size)

    # Specify number of training steps
    training_steps = 60000

    # Initialize TensorFlow monitored training session
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir = "./Model/Checkpoints/",
            hooks = [tf.train.StopAtStepHook(last_step=training_steps)],
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

        # Evaluate model on training dataset
        x_tdata, y_tdata = model.get_train_data()
        train_loss = model.evaluate(x_tdata, y_tdata)
        print("TRAINING LOSS = %.10f" %(train_loss))

        # Evaluate model on validation dataset
        x_vdata, y_vdata = model.get_val_data()
        val_loss = model.evaluate(x_vdata, y_vdata)
        print("VALIDATION LOSS = %.10f" %(val_loss))

        # Plot predictions
        model.plot_predictions()

# Run main() function when called directly
if __name__ == '__main__':
    main()
