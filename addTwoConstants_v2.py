import tensorflow as tf

s = tf.constant(2.0) + tf.constant(5.0)
sess = tf.Session()
sess.run(s)
