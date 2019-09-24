import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
import tqdm

from assignment_one.src.auxiliary import franke_function, mse, r2, create_variance_table
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

            regressor = Regressor(X_train, z_train)

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


if __name__ == '__main__':
    N = 1000

    polynomial_degrees = range(0, 6)
    cartesian_product = np.random.random(size=(N, 2))
    X, Y = np.meshgrid(cartesian_product[:, 0], cartesian_product[:, 1])
    z_grid = franke_function(X, Y)
    z_grid_noise = z_grid + np.random.normal(0, 0.05, size=(z_grid.shape))
    z = z_grid.ravel()
    z_noise = z_grid_noise.ravel()

    number_of_folds = 10

    mse_s, r2_s, beta, beta_variance, beta_std, feature_names = perform_cross_validation(cartesian_product, z_noise,
                                                                                         number_of_folds=number_of_folds,
                                                                                         Regressor=OLS)
    # plot_errors()
    variance_table = create_variance_table(beta_variance, feature_names, polynomial_degrees)
    latex_variance = variance_table.to_latex(float_format=r"num{{{:.3e}}}".format, na_rep='')
    # Illustration of how the variance increases with model complexity
    print(latex_variance)

    expectation_table = create_variance_table(beta, feature_names, polynomial_degrees)
    expectation_table = expectation_table.to_latex(float_format=r"num{{{:.3e}}}".format, na_rep='')
    print(expectation_table)
