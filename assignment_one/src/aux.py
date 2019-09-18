import numpy as np


def franke_function(x, y):
    """
    Evaluates the franke function over the mesh-grid X, Y.

    :param x: x values
    :param y: y values
    :return: F(x, y)
    """

    a = 3 / 4 * np.exp(-(9 * x - 2) ** 2 / 4 - (9 * y - 2) ** 2 / 4)
    b = 3 / 4 * np.exp(-(9 * x + 1) ** 2 / 49 - (9 * y + 1) / 10)
    c = 1 / 2 * np.exp(-(9 * x - 7) ** 2 / 4 - (9 * y - 3) ** 2 / 4)
    d = 1 / 5 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)

    return a + b + c - d


def mse(y, y_hat):
    """
    Computes the mean squared error of the prediction y_hat to the ground truth y.

    :param y: ground truth
    :param y_hat: prediction
    :return: mean squared error
    """

    n = y.shape[0]
    return ((y - y_hat) ** 2).sum() / n


def r2(y, y_hat):
    """
    Computes the r2-score (or coefficient of determination) of the prediction y_hat to the ground truth y.

    :param y: ground truth
    :param y_hat: prediction
    :return: r2-score
    """

    y_mean = y.mean()

    num = ((y - y_hat) ** 2).sum()
    den = ((y - y_mean) ** 2).sum()

    return 1 - num / den


def generate_data(N, noise=True, seed=None, return_mesh=False):
    """
    Samples (optionally noisy) data from the franke function with n parameter values in
    each parameter direction.

    :param N: number of values to sample in each parameter direction
    :param noise: whether to sample with added noise or not
    :param seed: optional seed for the RNG
    :return: x-values, y-values and F(x,y)-values in raveled arrays.
    """
    if noise and seed is not None:
        np.random.seed(seed)

    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)

    X, Y = np.meshgrid(x, y)
    Z = franke_function(X, Y).ravel()
    params = np.dstack((X, Y)).reshape(-1, 2)

    if noise:
        Z += np.random.normal(0, 1 / N, size=Z.shape)

    if return_mesh:
        return params, Z, X, Y
    else:
        return params, Z


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)

    X, Y = np.meshgrid(x, y)
    Z = franke_function(X, Y)

    fig = plt.figure()
    axs = Axes3D(fig)

    axs.plot_surface(X, Y, Z)
    plt.show()
