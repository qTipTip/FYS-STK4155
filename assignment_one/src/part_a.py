import numpy as np
import sklearn.preprocessing
import matplotlib.pyplot as plt
import tqdm
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import KFold

from assignment_one.src.auxiliary import franke_function
from assignment_one.src.plotting import latexify, format_axes
from assignment_one.src.regression_class import OLS


def perform_k_fold(X, y, number_of_folds=1, regressor=OLS):
    """
    Perform a single cross-validation resampling for the given regressor. All relevant
    results are returned in the dictionary results.

    :param X:
    :param number_of_folds:
    :param regressor:
    :param seed:
    :return:
    """
    results = {}
    kf = KFold(n_splits=number_of_folds, shuffle=True)

    for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        print(i, train_idx, test_idx)
        return results


def perform_ols_estimates_with_and_without_noise(seed=42, number_of_folds=1):
    np.random.seed(seed)
    polynomial_degrees = range(2, 11)

    N = 100
    signal_to_noise = 0.01
    x = np.random.random((N, 2))

    z = np.array([franke_function(X, Y) for X, Y in x])
    z_noise = z + signal_to_noise * np.random.normal(0, 1, z.shape)

    regressor = OLS
    for name, data in zip(['true', 'noise'], [z, z_noise]):
        for d in polynomial_degrees:
            poly_fit = sklearn.preprocessing.PolynomialFeatures(degree=d)
            X = poly_fit.fit_transform(x)
            results = perform_k_fold(X, data, number_of_folds, regressor)

    return


if __name__ == '__main__':
    perform_ols_estimates_with_and_without_noise(42, 10)
