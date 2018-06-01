import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Create artificial data 
x_data = np.pi/2 * np.random.normal(size=[100*10000, 1])
y_data = np.sin(x_data)

# Define loader for training dataset with mini-batch size 100
def initialize_loader():
    dataset = tf.data.Dataset.from_tensor_slices((x_data,y_data))
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(100*5))
    dataset = dataset.batch(100)
    dataset = dataset.prefetch(100*5)
    dataset = dataset.make_one_shot_iterator()
    dataset = dataset.get_next()
    return dataset

# Define placeholders for input and ouput values
x = tf.placeholder(tf.float32, [None, 1], name='x')
y = tf.placeholder(tf.float32, [None, 1], name='y')

# Define placeholder for learning rate
learning_rt = tf.placeholder(tf.float32, name='learning_rt')

# Define fully-connected layer with 10 hidden units
h = tf.layers.dense(x, 10, activation=tf.nn.relu)

# Define fully-connected layer with 20 hidden units
h = tf.layers.dense(h, 20, activation=tf.nn.relu)

# Define fully-connected layer with 10 hidden units
h = tf.layers.dense(h, 10, activation=tf.nn.relu)

# Define fully-connected layer to single ouput prediction
pred = tf.layers.dense(h, 1, activation=None)

# Define loss function
loss = tf.reduce_mean(tf.pow(pred - y, 2))

# Define optimizer
optim = tf.train.AdamOptimizer(learning_rate=learning_rt).minimize(loss)

# Define variable initializer
init = tf.global_variables_initializer()

# Initialize TensorFlow session
with tf.Session() as sess:

    # Initialize variables
    sess.run(init)

    # Initialize data loader
    dataset = initialize_loader()

    # Specify initial learning rate
    learning_rate = 0.001
    
    # Iterate through 20000 training steps
    for n in range(0,20000):

        # Retrieve batch from data loader
        x_batch, y_batch = sess.run(dataset)

        # Apply decay to learning rate every 1000 steps
        if n % 1000 == 0:
            learning_rate = 0.9*learning_rate
        
        # Run optimization operation for current mini-batch
        fd = {x: x_batch, y: y_batch, learning_rt: learning_rate}
        sess.run(optim, feed_dict=fd)

    # Plot predicted and true values for qualitative evaluation
    eval_pts = np.expand_dims(np.linspace(-np.pi/2, np.pi/2, 1000) , 1)
    predictions = sess.run(pred, feed_dict={x: eval_pts})
    true_values = np.sin(eval_pts)
    plt.plot(eval_pts[:,0], predictions[:,0], 'b')
    plt.plot(eval_pts[:,0], true_values[:,0], 'r')
    plt.show()
