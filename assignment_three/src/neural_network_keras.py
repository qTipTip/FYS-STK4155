import numpy as np
import tensorflow as tf

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

placeholder_zeros = tf.reshape(tf.convert_to_tensor(np.zeros(t.shape), dtype=tf.float64), shape=(-1, 1))

def initial_condition(x):
    return tf.sin(np.pi * x)


def trial_loss(dnn_output, gt=None):
    g_trial = (1 - t) * initial_condition(x) + x * (1 - x) * t * dnn_output
    g_trial_dt = tf.gradients(g_trial, t)
    g_trial_dx_dx = tf.gradients(tf.gradients(g_trial, x), x)
    return tf.losses.mean_squared_error(placeholder_zeros, g_trial_dt[0] - g_trial_dx_dx[0])



# Construction
t_tf = tf.convert_to_tensor(t.reshape((-1, 1)), dtype=tf.float64)  # x-values
x_tf = tf.convert_to_tensor(x.reshape((-1, 1)), dtype=tf.float64)  # t-values
p_tf = tf.convert_to_tensor(xt, dtype=tf.float64)  # (x, t)-values

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(20, activation='relu')
])

model.compile(optimizer='adam', loss=trial_loss)
model.fit(p_tf)
