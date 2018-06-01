# Initialize TensorFlow session
with tf.Session() as sess:

    # Specify values to feed into placeholder 'x'
    fd = { x : [1.,2.,3.] }
    
    # Run operation 'tf.add'
    y_vals = sess.run(y, feed_dict=fd)
