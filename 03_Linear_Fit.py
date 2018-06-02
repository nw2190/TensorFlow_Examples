import tensorflow as tf
import numpy as np

# Define placeholders for input and ouput values
x = tf.placeholder(tf.float32, [None, 1], name='x')
y = tf.placeholder(tf.float32, [None, 1], name='y')

# Define initializer for variables
var_init = tf.truncated_normal_initializer()

# Define trainable variable for slope
m = tf.get_variable("slope", dtype=tf.float32, shape=[1],
                    initializer=var_init, trainable=True)

# Define trainable variable for intercept
b = tf.get_variable("intercept", dtype=tf.float32, shape=[1],
                    initializer=var_init, trainable=True)

# Define predictions using slope and intercept variables
pred = tf.add(tf.multiply(m,x), b)

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
        x_batch = np.random.normal(size=[100, 1])
        y_batch = 4.0 * x_batch + 2.0

        # Run optimization operation for current mini-batch
        fd = {x: x_batch, y: y_batch}
        sess.run(optim, feed_dict=fd)

    # Plot predicted and true values for qualitative evaluation
    pred_m, pred_b = sess.run([m, b])
    print(pred_m)
    print(pred_b)
