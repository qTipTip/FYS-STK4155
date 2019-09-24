import numpy as np
import sklearn

import matplotlib.pyplot as plt
from assignment_one.src.auxiliary import franke_function, perform_cross_validation, create_variance_table, mse, r2
from assignment_one.src.regression_class import Ridge, Lasso, OLS

if __name__ == '__main__':
    N = 10000
    d = 5

    np.random.seed(42)

    lambda_vals = np.logspace(-5, 0, 50)

    cartesian_product = np.random.random(size=(N, 2))
    X, Y = np.meshgrid(cartesian_product[:, 0], cartesian_product[:, 1])
    z_grid = franke_function(X, Y)
    z_grid_noise = z_grid + np.random.normal(0, 0.5, size=(z_grid.shape))
    z = z_grid.ravel()
    z_noise = z_grid_noise.ravel()

    number_of_folds = 10

    results = {}
    for i, (name, regressor) in enumerate(zip(['Ridge', 'Lasso'], [Ridge, Lasso])):

        mse_results = []
        r2_results = []
        for l in lambda_vals:
            kf = sklearn.model_selection.KFold(n_splits=number_of_folds, shuffle=True)

            mse_point_vals = []
            r2_point_vals = []

            for train_idx, test_idx in kf.split(cartesian_product):
                x_train, x_test = cartesian_product[train_idx], cartesian_product[test_idx]
                z_train, z_test = z_noise[train_idx], z_noise[test_idx]

                poly_fit = sklearn.preprocessing.PolynomialFeatures(degree=d)
                X_train = poly_fit.fit_transform(x_train)
                X_test = poly_fit.fit_transform(x_test)

                R = regressor(X_train, z_train, lmbd=l, inversion_method='svd')

                z_hat_train = R.predict(X_train)
                z_hat_test = R.predict(X_test)

                mse_point_vals.append((mse(z_train, z_hat_train), mse(z_test, z_hat_test)))
                r2_point_vals.append((r2(z_train, z_hat_train), r2(z_test, z_hat_test)))

            mse_results.append(np.array(mse_point_vals).mean(axis=0))
            r2_results.append(np.array(r2_point_vals).mean(axis=0))

        mse_results = np.array(mse_results)
        r2_results = np.array(r2_results)

        results[name] = {'lambda': lambda_vals, 'mse': mse_results, 'r2': r2_results}

        # plt.subplot(2, 1, i+1)
        plt.loglog(results[name]['lambda'], results[name]['mse'][:, 0], label=name)
        # plt.loglog(results[name]['lambda'], results[name]['mse'][:, 1])
    plt.legend()
    plt.show()
