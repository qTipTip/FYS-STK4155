import numpy as np
import sklearn.preprocessing
import matplotlib.pyplot as plt
import tqdm
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import KFold

from assignment_one.src.auxiliary import franke_function, mse, r2
from assignment_one.src.plotting import latexify, format_axes
from assignment_one.src.regression_class import OLS, Ridge


def perform_k_fold(X, y, number_of_folds=1, regressor=OLS, regressor_parameters={}):
    """
    Perform a single cross-validation resampling for the given regressor. All relevant
    intermediate_results are returned in the dictionary intermediate_results.

    :param X:
    :param number_of_folds:
    :param regressor:
    :param seed:
    :return:
    """
    intermediate_results = {'mse': [], 'r2': [], 'model_bias': [], 'model_variance': [], 'beta_variance': []}

    kf = KFold(n_splits=number_of_folds, shuffle=True)

    for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_test, X_train = X[test_idx], X[train_idx]
        y_test, y_train = y[test_idx], y[train_idx]

        regr = regressor(X_train, y_train, **regressor_parameters)

        y_hat_train = regr.predict(X_train)
        y_hat_test = regr.predict(X_test)
        test_pred_mean = y_hat_test.mean()

        intermediate_results['mse'].append(mse(y_test, y_hat_test))
        intermediate_results['r2'].append(r2(y_test, y_hat_test))
        intermediate_results['model_variance'].append(np.mean((y_hat_test - test_pred_mean)) ** 2)
        intermediate_results['model_bias'].append(np.mean((y_test - test_pred_mean) ** 2))
        intermediate_results['beta_variance'].append(regr.beta_variance_estimate)

    intermediate_results.update({k: np.array(v).mean(axis=0) for (k, v) in intermediate_results.items()})

    return intermediate_results


def perform_ols_estimates_with_and_without_noise(seed=42, number_of_folds=1):
    np.random.seed(seed)
    polynomial_degrees = range(0, 5)

    N = 100
    signal_to_noise = 0.01
    x = np.random.random((N, 2))

    z = np.array([franke_function(X, Y) for X, Y in x])
    z_noise = z + signal_to_noise * np.random.normal(0, 1, z.shape)

    regressor = Ridge
    regr_params = {'lmbd': 0.001}
    for name, data in zip(['true', 'noise'], [z, z_noise]):
        for d in polynomial_degrees:
            poly_fit = sklearn.preprocessing.PolynomialFeatures(degree=d)
            X = poly_fit.fit_transform(x)
            results = perform_k_fold(X, data, number_of_folds, regressor, regr_params)


    return


if __name__ == '__main__':
    perform_ols_estimates_with_and_without_noise(42, 10)
