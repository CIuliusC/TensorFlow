import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

M     = 25                           # --- Number of harmonics
N     = 200                          # --- Number of frequency sampling points

wp    = 0.4 * np.pi                  # --- Maximum angular frequency

# --- Angular frequencies
omega = np.arange(0, np.pi, np.pi / N, dtype = np.float32)
# --- Harmonics
m     = np.arange(0, M, dtype = np.float32)

# --- Setting desired filter response
y_target = np.zeros((N, 1), dtype = np.float32)
for i in range(N):
  if omega[i]<wp:
    y_target[i]=1
  else:
    y_target[i]=0
  
# --- System matrix
MM, OMEGA = np.meshgrid(m, omega)
G = np.cos(MM * OMEGA)

# --- Tolerance
epsilon = tf.constant([0.00001])

########################
# OBJECTIVE FUNCTIONAL #
########################
def errorFunctional(w):
  #value = tf.reduce_mean(tf.maximum(0., tf.subtract(tf.abs(tf.subtract(tf.matmul(G, w), tf.abs(y_target))), epsilon))) + 0.5 * tf.reduce_mean(tf.abs(w) * tf.abs(w))
  value = tf.reduce_mean(tf.maximum(0., tf.subtract(tf.abs(tf.subtract(tf.matmul(G, w), y_target)), epsilon)))
  return value, tf.gradients(value, w)[0]

# --- Initial guess
initialGuess = tf.zeros(shape = [M, 1], dtype = tf.float32)
optim_results = tfp.optimizer.bfgs_minimize(errorFunctional, initial_position = initialGuess, tolerance = 1e-8)

with tf.Session() as session:
  results = session.run(optim_results)
  print("Convergence: %s" % results.converged)
  print("Failed: %s" % results.failed)
  print("Function evaluations: %d" % results.num_objective_evaluations)

  plt.plot(omega, np.matmul(G, results.position))
  plt.xlabel('$\omega$', fontsize = 14)
  plt.ylabel('Filter response', fontsize = 14)
