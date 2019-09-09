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
