import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

for N in [10, 20, 30, 40, 100]:
    tf.reset_default_graph()
    x_np = np.linspace(0, 1, N)

    # Nt = 10
    t_np = np.linspace(0, 1, N)

    X, T = np.meshgrid(x_np, t_np)

    x = X.ravel()
    t = T.ravel()

    ## The construction phase

    zeros = tf.reshape(tf.convert_to_tensor(np.zeros(x.shape)), shape=(-1, 1))
    x = tf.reshape(tf.convert_to_tensor(x), shape=(-1, 1))
    t = tf.reshape(tf.convert_to_tensor(t), shape=(-1, 1))

    points = tf.concat([x, t], 1)

    num_iter = 10000
    num_hidden_neurons = [20, 20, 20]

    X = tf.convert_to_tensor(X)
    T = tf.convert_to_tensor(T)

    with tf.variable_scope('dnn'):
        num_hidden_layers = np.size(num_hidden_neurons)

        previous_layer = points

        for l in range(num_hidden_layers):
            current_layer = tf.layers.dense(previous_layer, num_hidden_neurons[l], activation=tf.nn.sigmoid)
            previous_layer = current_layer

        dnn_output = tf.layers.dense(previous_layer, 1)


    def initial_conditions(x):
        return tf.sin(np.pi * x)


    with tf.name_scope('loss'):
        g_trial = (1 - t) * initial_conditions(x) + x * (1 - x) * t * dnn_output

        g_trial_dt = tf.gradients(g_trial, t)
        g_trial_d2x = tf.gradients(tf.gradients(g_trial, x), x)

        loss = tf.losses.mean_squared_error(zeros, g_trial_dt[0] - g_trial_d2x[0])

    learning_rate = 0.01
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        traning_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    exp = tf.exp(-np.pi ** 2 * t)
    sin = tf.sin(np.pi * x)
    g_analytic = exp * sin
    g_dnn = None

    ## The execution phase
    with tf.Session() as sess:
        init.run()
        for i in range(num_iter):
            sess.run(traning_op)

            # If one desires to see how the cost function behaves during training
            if i % 1000 == 0:
                print(loss.eval())

        g_analytic = g_analytic.eval()
        g_dnn = g_trial.eval()

    ## Compare with the analutical solution
    G_analytic = g_analytic.reshape((N, N))
    G_dnn = g_dnn.reshape((N, N))
    error = np.abs(g_analytic - g_dnn)


    print(f"""
    Errors with dx = {1 / N:.03f}
        1-norm = {np.max(np.abs(error)):.5f}
        2-norm = {np.linalg.norm(error, ord=2):.5f}
    """)

#
# # Plot the results
#
# X, T = np.meshgrid(x_np, t_np)
#
# fig = plt.figure(figsize=(10, 10))
# ax = fig.gca(projection='3d')
# ax.set_title('Solution from the deep neural network w/ %d layer' % len(num_hidden_neurons))
# s = ax.plot_surface(X, T, G_dnn, linewidth=0, antialiased=False, cmap=cm.viridis)
# ax.set_xlabel('Time $t$')
# ax.set_ylabel('Position $x$');
#
# fig = plt.figure(figsize=(10, 10))
# ax = fig.gca(projection='3d')
# ax.set_title('Analytical solution')
# s = ax.plot_surface(X, T, G_analytic, linewidth=0, antialiased=False, cmap=cm.viridis)
# ax.set_xlabel('Time $t$')
# ax.set_ylabel('Position $x$');
#
# fig = plt.figure(figsize=(10, 10))
# ax = fig.gca(projection='3d')
# ax.set_title('Difference')
# s = ax.plot_surface(X, T, diff, linewidth=0, antialiased=False, cmap=cm.viridis)
# ax.set_xlabel('Time $t$')
# ax.set_ylabel('Position $x$');
#
# ## Take some 3D slices
#
# indx1 = 0
# indx2 = int(Nt / 2)
# indx3 = Nt - 1
#
# t1 = t_np[indx1]
# t2 = t_np[indx2]
# t3 = t_np[indx3]
#
# # Slice the results from the DNN
# res1 = G_dnn[indx1, :]
# res2 = G_dnn[indx2, :]
# res3 = G_dnn[indx3, :]
#
# # Slice the analytical results
# res_analytical1 = G_analytic[indx1, :]
# res_analytical2 = G_analytic[indx2, :]
# res_analytical3 = G_analytic[indx3, :]
#
# # Plot the slices
# plt.figure(figsize=(10, 10))
# plt.title("Computed solutions at time = %g" % t1)
# plt.plot(x_np, res1)
# plt.plot(x_np, res_analytical1)
# plt.legend(['dnn', 'analytical'])
#
# plt.figure(figsize=(10, 10))
# plt.title("Computed solutions at time = %g" % t2)
# plt.plot(x_np, res2)
# plt.plot(x_np, res_analytical2)
# plt.legend(['dnn', 'analytical'])
#
# plt.figure(figsize=(10, 10))
# plt.title("Computed solutions at time = %g" % t3)
# plt.plot(x_np, res3)
# plt.plot(x_np, res_analytical3)
# plt.legend(['dnn', 'analytical'])
#
# plt.show()
