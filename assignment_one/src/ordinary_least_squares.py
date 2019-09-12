import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def design_matrix_from_parameters(params, polynomial_degree=2, intercept=True):
    """
    Constructs the design matrix consisting of polynomial basis functions evaluated at all data points.

    :param params: array of [x[i], y[i]] data points.
    :param polynomial_degree: degree of the polynomial fit
    :param intercept: whether to include a constant term
    :return: a vandermonde matrix
    """

    poly = PolynomialFeatures(degree=polynomial_degree, include_bias=intercept)
    return poly.fit_transform(params)


def ordinary_least_squares(X, y, pseudo_inv=True):
    """
    Solves the ordinary least squares problem by solving the normal equations X^T y =  X^T X b for b.

    :param pseudo_inv: uses an svd based approach for the inverse
    :param X: design matrix
    :param y: ground truth / response variable
    :return: the parameters b minimizing the mean squared error.
    """

    dot = X.T.dot(X)

    if y.ndim != 1:
        y = y.ravel()

    if pseudo_inv:
        dot_inv = svd_inv(dot)
    else:
        dot_inv = np.linalg.inv(dot)

    print(dot_inv.shape, X.T.shape, y.shape)
    b = dot_inv.dot(X.T).dot(y)
    return b


def ridge_regression(X, y, l=0.1, pseudo_inv=True):
    """
    Solves the ridge regression problem by solving the equations X^T y = (X^T X + LI)b for b.

    :param l: ridge hyper parameter
    :param X: design matrix
    :param y: ground truth / response variable
    :param pseudo_inv: whether to use the svd-based inverse
    :return: the parameters b minimizing the penalized mean squared error.
    """

    n = X.shape[1]
    dot = X.T.dot(X) + l * np.eye(n)
    if y.ndim != 1:
        y = y.ravel()

    if pseudo_inv:
        dot_inv = svd_inv(dot)
    else:
        dot_inv = np.linalg.inv(dot)

    print(dot_inv.shape, X.T.shape, y.shape)
    b = dot_inv.dot(X.T).dot(y)
    return b


def svd_inv(A):
    """
    Returns the inverse of the matrix A based on a singular value decomposition of A.

    :param A: matrix to invert
    :return: inv(A)
    """

    u, sigma, v_transposed = np.linalg.svd(A)
    m = u.shape[0]
    n = v_transposed.shape[0]

    D = np.zeros((m, n))
    for i in range(n):
        D[i, i] = sigma[i]

    return v_transposed.T @ (np.linalg.pinv(D) @ u.T)


if __name__ == '__main__':
    x = np.linspace(0, 1, 3)
    y = np.linspace(0, 1, 3) + np.random.normal(0, 1, 3)
    y_ = np.linspace(0, 1, 3)

    X = design_matrix_from_parameters(x, y, polynomial_degree=1)
    X_ = svd_inv(X)
    b = ordinary_least_squares(X, y_, pseudo_inv=False)


def perform_regression(x, y, z, polynomial_degree=5, ridge=False, l=0.1):
    """
    Performs a polynomial fit to sampled data.
    :return: parameters beta, and sampled points from predicted surface.

    """
    N, M = x.shape[0], y.shape[1]
    params = np.dstack((x, y)).reshape(-1, 2)
    n = int(np.sqrt(z.shape[0]))
    v = design_matrix_from_parameters(params, polynomial_degree=polynomial_degree)

    if ridge:
        beta = ridge_regression(v, z, l=l)
    else:
        beta = ordinary_least_squares(v, z)
    print(v.shape, beta.shape)
    z_hat = (v @ beta).reshape(N, M)

    return beta, z_hat
