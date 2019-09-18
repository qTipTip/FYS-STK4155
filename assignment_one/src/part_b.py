import numpy as np
import sklearn.preprocessing
import sklearn.model_selection
import tqdm
import matplotlib.pyplot as plt

from assignment_one.src.aux import franke_function, mse, r2
from assignment_one.src.plotting import latexify, format_axes
from assignment_one.src.regression_class import OLS

if __name__ == '__main__':
    N = 1000

    polynomial_degrees = range(0, 10)

    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)

    X, Y = np.meshgrid(x, y)
    cartesian_product = np.dstack((X, Y)).reshape(-1, 2)
    z_grid = franke_function(X, Y)
    z_grid_noise = z_grid + np.random.normal(0, 0.05, size=(z_grid.shape))
    z = z_grid.ravel()
    z_noise = z_grid_noise.ravel()

    number_of_folds = 5
    kf = sklearn.model_selection.KFold(n_splits=number_of_folds, shuffle=True)

    mse_s = []
    r2_s = []
    beta_variance = []
    beta_std = []
    for d in tqdm.tqdm(polynomial_degrees):
        mse_scores = []
        r2_scores = []
        bvar = []
        for train_idx, test_idx in kf.split(cartesian_product):
            x_train, x_test = cartesian_product[train_idx], cartesian_product[test_idx]
            z_train, z_test = z[train_idx], z[test_idx]

            X_train = sklearn.preprocessing.PolynomialFeatures(degree=d).fit_transform(x_train)
            X_test = sklearn.preprocessing.PolynomialFeatures(degree=d).fit_transform(x_test)

            ols = OLS(X_train, z_train)

            z_hat_train = ols.predict(X_train)
            z_hat_test = ols.predict(X_test)

            mse_scores.append((mse(z_train, z_hat_train), mse(z_test, z_hat_test)))
            r2_scores.append((r2(z_train, z_hat_train), r2(z_test, z_hat_test)))
            bvar.append(ols.beta_variance_estimate)
        mse_s.append(np.array(mse_scores).mean(axis=0))
        r2_s.append(np.array(r2_scores).mean(axis=0))
        beta_std.append(np.array(bvar).mean(axis=0))

    mse_s = np.array(mse_s)
    r2_s = np.array(r2_s)

    latexify(3)
    plt.semilogy(polynomial_degrees, mse_s[:,1], ls='-', marker='o')
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
