import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# --- Setting eager mode.
tfe.enable_eager_execution()

# --- Define constant tensors.
a = tf.constant(2)
b = tf.constant(3)
print("a = %i" % a)
print("b = %i" % b)

# --- Run the operation without the session.
c = a + b
d = a * b
print("a + b = %i" % c)
print("a * b = %i" % d)
