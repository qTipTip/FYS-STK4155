import numpy as np
from matplotlib import cm

from assignment_one.src.aux import generate_data, mean_squared_error, r2_score
from assignment_one.src.ordinary_least_squares import perform_regression, svd_inv


def compute_beta_variance(noise_variance, design_matrix):
    """
    Computes the variance of the model parameters beta.

    :param beta:
    :return:
    """

    return noise_variance * svd_inv(design_matrix.T.dot(design_matrix))


if __name__ == '__main__':

    import tqdm
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.model_selection import KFold

    poly_degs = range(6)
    N = 1500
    seed = 42
    noise = False
    params, z, X, Y = generate_data(N, noise, seed, return_mesh=True)

    kf = KFold(n_splits=10)
    for train_idx, test_idx in kf.split(X):
        for d in tqdm.tqdm(poly_degs, disable=True):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            z_train, z_test = z[train_idx], z[test_idx]

            print(X_train)
            beta, z_hat = perform_regression(X_train, Y_train, z_train, polynomial_degree=d, ridge=True, l=0.1)

            M = X_train.shape[0]
            z_hat = z_hat.reshape(M, M)
            z_train = z_train.reshape(M, M)

            fig = plt.figure()
            ax1 = fig.add_subplot(211, projection='3d')
            ax2 = fig.add_subplot(212, projection='3d')

            ax1.view_init(azim=30)
            ax2.view_init(azim=30)
            ax1.set_zlim3d(-0.2, 1.2)

            ax1.plot_surface(X, Y, z_hat, alpha=0.5, cmap=cm.coolwarm)
            # ax1.scatter(X, Y, z, alpha=1, s=1, color='black')

            ax2.plot_surface(X, Y, np.abs(z_hat - z), cmap=cm.coolwarm)
            ax2.set_zlim3d(0, 1)

            plt.title(f'Polynomial degree $p = {d}$')
            plt.show()

            z_hat = z_hat.ravel()
            z = z.ravel()

            print('Polynomial degree = ', d)
            print(f'\tMSE = {mean_squared_error(z, z_hat):.3f}')
            print(f'\tR2  = {r2_score(z, z_hat):.3f}')
            print(f'Predicted beta variance = {beta.var():.3f}')
            print(f'Computed beta variance  = {compute_beta_variance(1, X)}')
            print('===========================================')
