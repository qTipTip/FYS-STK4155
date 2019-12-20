import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
import numpy as np

Q = np.random.normal(0, 1, (6, 6))
A = (Q.T + Q) / 2
I = np.eye(6)
num_iter = 100000

for N in [6]:

    tf.reset_default_graph()
    x_np = np.random.random(N)
    # Nt = 10
    t_np = np.arange(num_iter, dtype=np.float64)

    ## The construction phase

    zeros = tf.reshape(tf.convert_to_tensor(np.zeros(x_np.shape)), shape=(-1, 1))
    x = tf.reshape(tf.convert_to_tensor(x_np), shape=(-1, 1))
    t = tf.reshape(tf.convert_to_tensor(t_np), shape=(-1, 1))

    points = tf.concat([x, t], 1)

    num_hidden_neurons = [20, 20, 20]

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
        u = dnn_output
        u_dt = tf.gradients(u, x)[0]

        f = tf.matmul(tf.transpose(u) * u * A + (1 - tf.transpose(u) * A * u) * I, u)

        loss = tf.losses.mean_squared_error(zeros, u_dt + u - f)

    learning_rate = 0.001
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        traning_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    g_dnn = None

    ## The execution phase
    with tf.Session() as sess:
        init.run()
        for i in range(num_iter):
            sess.run(traning_op)

            # If one desires to see how the cost function behaves during training
            if i % 1000 == 0:
                # print(loss.eval())
                print(u_dt.eval())
                print((u - f).eval())

        # g_analytic = g_analytic.eval()
        g_dnn = dnn_output

        v = g_dnn.eval()
        v = np.array(v)
        # print(v.T @ v, v.T @ A @ v)
        eigval = (v.T @ A @ v) / (v.T @ v)

        print(eigval, np.linalg.eigvals(A))

        # print('AV', np.dot(A, v), 'lambda', eigval, 'lambdav', eigval * v)
        # print(np.dot(A, v) - eigval * v)
    ## Compare with the analutical solution
    # G_analytic = g_analytic.reshape((N, N))
    # G_dnn = g_dnn.reshape((N, N))
    # error = np.abs(g_analytic - g_dnn)

    # print(f"""
    # Errors with dx = {1 / N:.03f}
    #     1-norm = {np.max(np.abs(error)):.5f}
    #     2-norm = {np.linalg.norm(error, ord=2):.5f}
    # """)

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
