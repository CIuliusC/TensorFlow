import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# --- Setting eager mode.
tfe.enable_eager_execution()


# --- Full compatibility with Numpy.
# --- Tensorflow tensor
a = tf.constant([[2., 1.],
                 [1., 0.]], dtype=tf.float32)
print("Tensor:\n a = %s" % a)
# --- Numpy array
b = np.array([[3., 0.],
              [5., 1.]], dtype=np.float32)
print("NumpyArray:\n b = %s" % b)

c = a + b
print("a + b = %s" % c)

d = tf.matmul(a, b)
print("a * b = %s" % d)
