from matplotlib import cm

from assignment_one.src import ordinary_least_squares as ols
from assignment_one.src.aux import generate_data, mean_squared_error, r2_score


def perform_regression(params, z, polynomial_degree=5, ridge=False, l=0.1):
    """
    Performs a polynomial fit to sampled data from the Franke function.
    :return: parameters beta, and sampled points from predicted surface.

    """
    n = int(np.sqrt(z.shape[0]))
    v = ols.design_matrix_from_parameters(params, polynomial_degree=polynomial_degree)

    if ridge:
        beta = ols.ridge_regression(v, z, l=l)
    else:
        beta = ols.ordinary_least_squares(v, z)
    z_hat = (v @ beta).reshape(n, n)

    return beta, z_hat


def compute_beta_variance(beta):
    """
    Computes the variance of the model parameters beta.

    :param beta:
    :return:
    """

    pass


if __name__ == '__main__':

    import tqdm
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    poly_degs = range(6)
    N = 50
    seed = 42
    noise = True
    params, z, X, Y = generate_data(N, noise, seed, return_mesh=True)

    for d in tqdm.tqdm(poly_degs, disable=True):
        beta, z_hat = perform_regression(params, z, polynomial_degree=d, ridge=True, l=0.1)

        z_hat = z_hat.reshape(N, N)
        z = z.reshape(N, N)

        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.set_zlim3d(-0.2, 1.2)

        ax1.plot_surface(X, Y, z_hat, alpha=0.5, cmap=cm.coolwarm)
        ax1.scatter(X, Y, z, alpha=1, s=1, color='black')

        plt.title(f'Polynomial degree $p = {d}$')
        plt.show()

        z_hat = z_hat.ravel()
        z = z.ravel()

        print('Polynomial degree = ', d)
        print(f'\tMSE = {mean_squared_error(z, z_hat):.3f}')
        print(f'\tR2  = {r2_score(z, z_hat):.3f}')
        print(f'Predicted beta variance = {beta.var():.3f}')
        print('===========================================')
