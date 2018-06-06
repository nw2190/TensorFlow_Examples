import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Class representation of network model
class Model(object):
    
    # Initialize model
    def __init__(self, sess, x_data, y_data, batch_size):
        self.sess = sess
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size

    # Define loader for training dataset with mini-batch size 100
    def initialize_loader(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.x_data,self.y_data))
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(self.batch_size*5))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.batch_size*5)
        dataset = dataset.make_one_shot_iterator()
        dataset = dataset.get_next()
        return dataset

    # Define graph for model
    def build_model(self):

        # Define placeholders for input and ouput values
        self.x = tf.placeholder(tf.float32, [None, 1], name='x')
        self.y = tf.placeholder(tf.float32, [None, 1], name='y')

        # Define placeholder for learning rate
        self.learning_rt = tf.placeholder(tf.float32, name='learning_rt')

        # Define fully-connected layer with 10 hidden units
        h = tf.layers.dense(self.x, 10, activation=tf.nn.relu)

        # Define fully-connected layer with 20 hidden units
        h = tf.layers.dense(h, 20, activation=tf.nn.relu)

        # Define fully-connected layer with 10 hidden units
        h = tf.layers.dense(h, 10, activation=tf.nn.relu)

        # Define fully-connected layer to single ouput prediction
        self.pred = tf.layers.dense(h, 1, activation=None)

        # Define loss function
        self.loss = tf.reduce_mean(tf.pow(self.pred - self.y, 2))

        # Define optimizer
        self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rt).minimize(self.loss)

        # Define variable initializer
        self.init = tf.global_variables_initializer()

    # Train model
    def train(self):

        # Initialize variables
        self.sess.run(self.init)

        # Initialize data loader
        self.dataset = self.initialize_loader()

        # Specify initial learning rate
        learning_rate = 0.001

        # Iterate through 20000 training steps
        for n in range(0,20000):

            # Retrieve batch from data loader
            x_batch, y_batch = self.sess.run(self.dataset)

            # Apply decay to learning rate every 1000 steps
            if n % 1000 == 0:
                learning_rate = 0.9*learning_rate

            # Run optimization operation for current mini-batch
            fd = {self.x: x_batch, self.y: y_batch, self.learning_rt: learning_rate}
            self.sess.run(self.optim, feed_dict=fd)

    # Define method for computing model predictions
    def predict(self, eval_pts):
        return self.sess.run(self.pred, feed_dict={self.x: eval_pts})
            
    # Plot predicted and true values for qualitative evaluation
    def evaluate(self):
        eval_pts = np.expand_dims(np.linspace(-np.pi/2, np.pi/2, 1000) , 1)
        predictions = self.predict(eval_pts)
        true_values = np.sin(eval_pts)
        plt.plot(eval_pts[:,0], predictions[:,0], 'b')
        plt.plot(eval_pts[:,0], true_values[:,0], 'r')
        plt.show()


# Initialize and train model 
def main():

    # Create artificial data 
    x_data = np.pi/2 * np.random.normal(size=[100*10000, 1])
    y_data = np.sin(x_data)

    # Initialize TensorFlow session
    with tf.Session() as sess:

        # Initialize model
        model = Model(sess, x_data, y_data, 100)

        # Build model graph
        model.build_model()

        # Train model
        model.train()

        # Evaluate model
        model.evaluate()

        
# Run main() function when called directly
if __name__ == '__main__':
    main()
