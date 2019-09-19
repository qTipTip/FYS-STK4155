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