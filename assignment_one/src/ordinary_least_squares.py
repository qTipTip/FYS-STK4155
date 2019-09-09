import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def design_matrix_from_parameters(x, y, polynomial_degree=2, intercept=True):
    """
    Constructs the design matrix consisting of polynomial basis functions evaluated at all data points.

    :param x: x-parameters
    :param y: y-parameters
    :param polynomial_degree: degree of the polynomial fit
    :param intercept: whether to include a constant term
    :return: a vandermonde matrix
    """

    X = np.stack((x, y), axis=1)
    poly = PolynomialFeatures(degree=polynomial_degree, include_bias=intercept)
    return poly.fit_transform(X)


if __name__ == '__main__':
    x = np.linspace(0, 1, 3)
    y = np.linspace(0, 1, 3)

    X = design_matrix_from_parameters(x, y)
    X_ = design_matrix_from_parameters(x, y, intercept=False)
