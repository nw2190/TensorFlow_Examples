import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Define placeholders for input and ouput values
x = tf.placeholder(tf.float32, [None, 1], name='x')
y = tf.placeholder(tf.float32, [None, 1], name='y')

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
optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Define variable initializer
init = tf.global_variables_initializer()

# Initialize TensorFlow session
with tf.Session() as sess:

    # Initialize variables
    sess.run(init)
    
    # Iterate through 10000 training steps
    for n in range(0,10000):

        # Create artificial data with mini-batch size 100
        x_batch = np.pi/2 * np.random.normal(size=[100, 1])
        y_batch = np.sin(x_batch)

        # Run optimization operation for current mini-batch
        fd = {x: x_batch, y: y_batch}
        sess.run(optim, feed_dict=fd)

    # Plot predicted and true values for qualitative evaluation
    eval_pts = np.expand_dims(np.linspace(-np.pi/2, np.pi/2, 1000) , 1)
    predictions = sess.run(pred, feed_dict={x: eval_pts})
    true_values = np.sin(eval_pts)
    plt.plot(eval_pts[:,0], predictions[:,0], 'b')
    plt.plot(eval_pts[:,0], true_values[:,0], 'r')
    plt.show()
