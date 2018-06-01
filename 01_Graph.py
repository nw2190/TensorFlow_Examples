import tensorflow as tf

# Define a placeholder for input values
x = tf.placeholder(tf.float32, [None], name='x')

# Define a constant value used to shift input values
shift = tf.constant(10.0, dtype=tf.float32, name="shift")

# Compute shifted values
y = tf.add(x, shift, name="y")
