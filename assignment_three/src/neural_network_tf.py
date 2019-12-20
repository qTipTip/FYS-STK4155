import numpy as np
import tensorflow.compat.v1 as tf

# from tensorflow_core.python import set_random_seed, global_variables_initializer, Session
# from tensorflow_core.python.framework.ops import disable_eager_execution
# from tensorflow_core.python.layers import layers
# from tensorflow_core.python.training.gradient_descent import GradientDescentOptimizer

tf.disable_eager_execution()


def initial_condition(x):
    return tf.sin(np.pi * x)


tf.set_random_seed(4155)

max_space = 1
num_space = 10
num_time = 10
max_time = 1
num_epochs = 10000
t_np = np.linspace(0, max_time, num_time)
x_np = np.linspace(0, max_space, num_space)

X, T = np.meshgrid(x_np, t_np)
xt = np.dstack((X, T)).reshape(-1, 2)
x, t = xt[:, 0], xt[:, 1]

# Construction
t_tf = tf.convert_to_tensor(t.reshape((-1, 1)), dtype=tf.float64)  # x-values
x_tf = tf.convert_to_tensor(x.reshape((-1, 1)), dtype=tf.float64)  # t-values
p_tf = tf.convert_to_tensor(xt, dtype=tf.float64)  # (x, t)-values

placeholder_zeros = tf.reshape(tf.convert_to_tensor(np.zeros(t.shape)), shape=(-1, 1))

num_hidden_neurons = [20, 20, 20]
num_hidden_layers = np.size(num_hidden_neurons)

# We define the dense neural network, with output 'dnn_output'.
with tf.name_scope('dnn'):
    previous_layer = p_tf

    for l in range(num_hidden_layers):
        current_layer = tf.layers.dense(previous_layer, num_hidden_neurons[l], name=f'hidden{l}')
        previous_layer = current_layer

    dnn_output = tf.layers.dense(previous_layer, 1, name='output')

# The cost function is defined in terms of the trial function.
with tf.name_scope('cost'):
    g_trial = (1 - t) * initial_condition(x) + x * (1 - x) * t * dnn_output
    g_trial_dt = tf.gradients(g_trial, t)
    g_trial_dx_dx = tf.gradients(tf.gradients(g_trial, x), x)
    loss = tf.losses.mean_squared_error(placeholder_zeros, g_trial_dt[0] - g_trial_dx_dx[0])

learning_rate = 0.01
with tf.name_scope('train'):
    optim = tf.train.GradientDescentOptimizer(learning_rate)
    training_optim = optim.minimize(loss)

init = tf.global_variables_initializer()
g_analytic = tf.exp(-np.pi ** 2 * 2) + tf.sin(np.pi * x)
g_dnn = None

with tf.Session() as sess:
    init.run()
    for i in range(num_epochs):
        print(g_trial.eval())
        sess.run(training_optim)

        if i % 10000 == 0:
            print(loss.eval())

    g_analytic = g_analytic.eval()
    g_dnn = g_trial.eval()

diff = np.abs(g_analytic - g_dnn)
G_analytic = g_analytic.reshape((num_time, num_space))
G_dnn = g_analytic.reshape((num_time, num_space))

Diff = np.abs(G_analytic - G_dnn)
print(diff, Diff)