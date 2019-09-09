import numpy as np

from assignment_one.src import ordinary_least_squares as ols
from assignment_one.src.aux import generate_data, mean_squared_error, r2_score


def perform_franke_ols(params, z, polynomial_degree=5):
    """
    Performs a polynomial fit to sampled data from the Franke function.
    :return: parameters beta, and sampled points from predicted surface.

    """
    n = int(np.sqrt(z.shape[0]))
    v = ols.design_matrix_from_parameters(params, polynomial_degree=polynomial_degree)
    beta = ols.ordinary_least_squares(v, z)
    z_hat = (v @ beta).reshape(n, n)

    return beta, z_hat


def compute_beta_variance(beta):
    """
    Computes the variance of the model parameters beta.

    :param beta:
    :return:
    """


if __name__ == '__main__':

    import tqdm

    poly_degs = range(6)
    N = 2000
    seed = 42
    noise = True
    params, z = generate_data(N, noise, seed)

    for d in tqdm.tqdm(poly_degs, disable=True):
        beta, z_hat = perform_franke_ols(params, z, polynomial_degree=d)

        z_hat = z_hat.ravel()
        z = z.ravel()

        print('Polynomial degree = ', d)
        print(f'\tMSE = {mean_squared_error(z, z_hat):.3f}')
        print(f'\tR2  = {r2_score(z, z_hat):.3f}')
        np.var
        print(f'Predicted beta variance = {beta.var():.3f}')
        print('===========================================')
