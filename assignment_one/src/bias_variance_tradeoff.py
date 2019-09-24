import numpy as np
import sklearn.model_selection
import tqdm as tqdm
import matplotlib.pyplot as plt

from assignment_one.src.auxiliary import franke_function, mse, r2
from assignment_one.src.plotting import latexify, format_axes
from assignment_one.src.regression_class import OLS

if __name__ == '__main__':
    N = 100
    NUMBER_OF_FOLDS = 10
    SIGNAL_TO_NOISE = 0.05

    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)

    cartesian_product = np.dstack((X, Y)).reshape(-1, 2)

    z_values = franke_function(X, Y).ravel() + SIGNAL_TO_NOISE * np.random.normal(0, 1, size=(N ** 2))

    polynomial_degrees = range(0, 16)

    kf = sklearn.model_selection.KFold(NUMBER_OF_FOLDS)

    mse_scores_polynomial = np.zeros((2, len(polynomial_degrees)))
    r2_scores_polynomial = np.zeros((2, len(polynomial_degrees)))

    bias = []
    vars = []

    for d in polynomial_degrees:

        print(f'Performing {NUMBER_OF_FOLDS}-fold cross-validation for degree {d} polynomial')
        # create design matrix for the given degree
        design_matrix = sklearn.preprocessing.PolynomialFeatures(degree=d).fit_transform(cartesian_product)

        # mse_values[0] is train and mse_values[1] is test scores
        # r2_values[0] is train and r2_values[1] is test scores
        mse_values = np.zeros((2, NUMBER_OF_FOLDS))
        r2_values = np.zeros((2, NUMBER_OF_FOLDS))

        # split into test and train
        test_predictions = []
        for i, (train_idx, test_idx) in enumerate(kf.split(design_matrix)):
            X_train, X_test = design_matrix[train_idx], design_matrix[test_idx]
            z_train, z_test = z_values[train_idx], z_values[test_idx]

            ols = OLS(X_train, z_train, inversion_method='svd')

            z_hat_train = ols.predict(X_train)
            z_hat_test = ols.predict(X_test)

            mse_values[:, i] = [mse(z_train, z_hat_train), mse(z_test, z_hat_test)]
            r2_values[:, i] = [r2(z_train, z_hat_train), r2(z_test, z_hat_test)]

            test_predictions.append(z_hat_test)

        test_predictions = np.array(test_predictions)

        # compute the mean values over all folds
        mse_scores_polynomial[:, d] = mse_values.mean(axis=1)
        r2_scores_polynomial[:, d] = r2_values.mean(axis=1)

        bias.append(np.mean(np.mean(test_predictions)))
    # plot estimated errors
    latexify(fig_width=4)
    plt.plot(polynomial_degrees, mse_scores_polynomial[0], ls='-', marker='o',
             label=r'$\mathrm{MSE}_{\mathrm{train}}(\mathbf{y}, \hat{\mathbf{y}})$')
    plt.plot(polynomial_degrees, mse_scores_polynomial[1], ls='--', marker='*',
             label=r'$\mathrm{MSE}_{\mathrm{test}}(\mathbf{y}, \hat{\mathbf{y}})$')
    plt.xlabel(r'$d$')
    plt.legend()
    plt.tight_layout()

    format_axes(plt.gca())

    plt.savefig('../article/images/OLS_MSE_score_crossval.pdf')
    plt.show()

    latexify(fig_width=4)
    plt.plot(polynomial_degrees, r2_scores_polynomial[0], ls='-', marker='o',
             label=r'$\mathrm{R}^2_{\mathrm{train}}(\mathbf{y}, \hat{\mathbf{y}})$')
    plt.plot(polynomial_degrees, r2_scores_polynomial[1], ls='--', marker='*',
             label=r'$\mathrm{R}^2_{\mathrm{test}}(\mathbf{y}, \hat{\mathbf{y}})$')
    plt.xlabel(r'$d$')
    plt.legend()
    plt.tight_layout()

    format_axes(plt.gca())
    plt.savefig('../article/images/OLS_R2_score_crossval.pdf')
    plt.show()
