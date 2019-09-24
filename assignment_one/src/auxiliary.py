import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.preprocessing
import tqdm

from assignment_one.src.part_b import polynomial_degrees
from assignment_one.src.regression_class import OLS


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


def format_feature_names(feature_names):
    feature_names = list(map(lambda x: x.replace('x0', 'x'), feature_names))
    feature_names = list(map(lambda x: x.replace('x1', 'y'), feature_names))
    feature_names = list(map(lambda x: x.replace('x^0y^0', '1'), feature_names))
    feature_names = list(map(lambda x: r'$' + x + r'$', feature_names))

    return feature_names


def create_variance_table(beta_variance, feature_names, polynomial_degrees):
    formatted_features = format_feature_names(feature_names)

    df = pd.DataFrame(columns=polynomial_degrees, index=formatted_features)
    for i, d in enumerate(polynomial_degrees):
        df[d][:len(beta_variance[i])] = beta_variance[i]

    return df


def perform_cross_validation(cartesian_product, z_values, number_of_folds=10, Regressor=OLS, lambd=None):
    mse_s = []
    r2_s = []
    beta_variance = []
    beta_std = []
    beta = []

    kf = sklearn.model_selection.KFold(n_splits=number_of_folds, shuffle=True)

    for d in tqdm.tqdm(polynomial_degrees):
        mse_scores = []
        r2_scores = []
        bvar = []
        std = []
        beta_vals = []
        for train_idx, test_idx in kf.split(cartesian_product):
            x_train, x_test = cartesian_product[train_idx], cartesian_product[test_idx]
            z_train, z_test = z_values[train_idx], z_values[test_idx]

            poly_fit = sklearn.preprocessing.PolynomialFeatures(degree=d)
            X_train = poly_fit.fit_transform(x_train)
            X_test = poly_fit.fit_transform(x_test)

            if lambd is None:
                regressor = Regressor(X_train, z_train)
            else:
                regressor = Regressor(X_train, z_train, lambd=lambd)

            z_hat_train = regressor.predict(X_train)
            z_hat_test = regressor.predict(X_test)

            mse_scores.append((mse(z_train, z_hat_train), mse(z_test, z_hat_test)))
            r2_scores.append((r2(z_train, z_hat_train), r2(z_test, z_hat_test)))
            bvar.append(regressor.beta_variance_estimate)
            std.append(regressor.beta_std_dev_estimate)
            beta_vals.append(regressor.beta)

        mse_s.append(np.array(mse_scores).mean(axis=0))
        r2_s.append(np.array(r2_scores).mean(axis=0))
        beta_variance.append(np.array(bvar).mean(axis=0))
        beta_std.append(np.array(std).mean(axis=0))
        beta.append(np.array(beta_vals).mean(axis=0))

    mse_s = np.array(mse_s)
    r2_s = np.array(r2_s)

    feature_names = poly_fit.get_feature_names()

    return mse_s, r2_s, beta, beta_variance, beta_std, feature_names
