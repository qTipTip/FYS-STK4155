import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection
import tqdm
import sympy
import matplotlib.pyplot as plt

from assignment_one.src.auxiliary import franke_function, mse, r2
from assignment_one.src.plotting import latexify, format_axes
from assignment_one.src.regression_class import OLS


def plot_errors():
    latexify(3)
    plt.semilogy(polynomial_degrees, mse_s[:, 1], ls='-', marker='o')
    plt.xlabel('$d$')
    plt.ylabel('$\\mathrm{MSE}(\\mathbf{y}, \\hat{\\mathbf{y}})$')
    plt.tight_layout()
    format_axes(ax=plt.gca())
    plt.savefig('../article/images/OLS_MSE_score_crossval.pdf')
    plt.clf()
    plt.plot(polynomial_degrees, r2_s[:, 1], ls='-', marker='o')
    plt.xlabel('$d$')
    plt.ylabel('$\\mathrm{R}^2(\\mathbf{y}, \\hat{\\mathbf{y}})$')
    plt.tight_layout()
    format_axes(ax=plt.gca())
    plt.savefig('../article/images/OLS_R2_score_crossval.pdf')


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


if __name__ == '__main__':
    N = 100

    polynomial_degrees = range(0, 5)

    cartesian_product = np.random.random(size=(N, 2))
    X, Y = np.meshgrid(cartesian_product[:, 0], cartesian_product[:, 1])
    # cartesian_product = np.dstack((X, Y)).reshape(-1, 2)
    z_grid = franke_function(X, Y)
    z_grid_noise = z_grid + np.random.normal(0, 0.05, size=(z_grid.shape))
    z = z_grid.ravel()
    z_noise = z_grid_noise.ravel()

    number_of_folds = 100
    kf = sklearn.model_selection.KFold(n_splits=number_of_folds, shuffle=True)

    mse_s = []
    r2_s = []
    beta_variance = []
    beta_std = []

    for d in tqdm.tqdm(polynomial_degrees):
        mse_scores = []
        r2_scores = []
        bvar = []
        std = []
        for train_idx, test_idx in kf.split(cartesian_product):
            x_train, x_test = cartesian_product[train_idx], cartesian_product[test_idx]
            z_train, z_test = z[train_idx], z[test_idx]

            poly_fit = sklearn.preprocessing.PolynomialFeatures(degree=d)
            X_train = poly_fit.fit_transform(x_train)
            X_test = poly_fit.fit_transform(x_test)

            ols = OLS(X_train, z_train)

            z_hat_train = ols.predict(X_train)
            z_hat_test = ols.predict(X_test)

            mse_scores.append((mse(z_train, z_hat_train), mse(z_test, z_hat_test)))
            r2_scores.append((r2(z_train, z_hat_train), r2(z_test, z_hat_test)))
            bvar.append(ols.beta_variance_estimate)
            std.append(ols.beta_std_dev_estimate)

        mse_s.append(np.array(mse_scores).mean(axis=0))
        r2_s.append(np.array(r2_scores).mean(axis=0))
        beta_variance.append(np.array(bvar).mean(axis=0))
        beta_std.append(np.array(std).mean(axis=0))

    feature_names = poly_fit.get_feature_names()

    mse_s = np.array(mse_s)
    r2_s = np.array(r2_s)

    # plot_errors()
    variance_table = create_variance_table(beta_variance, feature_names, polynomial_degrees)
    latex = variance_table.to_latex(float_format=r"num{{{:.3e}}}".format, na_rep='')
    print(mse_s)
    # Illustration of how the variance increases with model complexity
    print(latex)