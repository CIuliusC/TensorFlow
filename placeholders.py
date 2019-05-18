import tensorflow as tf

# --- Addition and multiplication with placeholders.
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.multiply(a, b)

sess = tf.Session()
print("Addition with placeholder       : %i" % sess.run(add, feed_dict={a: 5, b: 6}))
print("Multiplication with placeholder : %i" % sess.run(mul, feed_dict={a: 5, b: 6}))
