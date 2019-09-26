import numpy as np
import pytest
import sklearn

from assignment_one.src.auxiliary import r2
from assignment_one.src.regression_class import OLS, Lasso, Ridge


@pytest.mark.parametrize('inversion_method', ['svd'])
@pytest.mark.parametrize("d", range(10))
@pytest.mark.parametrize("regressor, sklearn_reg, lmd",
                         [(OLS, sklearn.linear_model.LinearRegression, None), (Ridge, sklearn.linear_model.Ridge, 0.1),
                          (Lasso, sklearn.linear_model.Lasso, 0.1)])
def test_against_sklearn(d, regressor, sklearn_reg, lmd, inversion_method):
    X, y = get_data(d)

    if lmd:
        linreg = sklearn_reg(fit_intercept=False, alpha=lmd).fit(X, y)
        O = regressor(X, y, inversion_method=inversion_method, lmbd=lmd)
    else:
        linreg = sklearn_reg(fit_intercept=False).fit(X, y)
        O = regressor(X, y, inversion_method=inversion_method)

    beta = O.beta
    y_hat = O.predict(X)

    beta_sklearn = linreg.coef_
    y_hat_sklearn = linreg.predict(X)

    np.testing.assert_allclose(beta, beta_sklearn, rtol=1e-4)
    np.testing.assert_allclose(y_hat, y_hat_sklearn, rtol=1e-4)

    sk_error = sklearn.metrics.mean_squared_error(y, y_hat_sklearn)
    error = O.mse()
    assert abs(sk_error - error) < 1.0e-4


def get_data(d):
    np.random.seed(42)
    X = np.array([
        [1, 1],
        [1, 2],
        [2, 2],
        [2, 3]
    ])
    X = sklearn.preprocessing.PolynomialFeatures(degree=d).fit_transform(X)
    y = np.dot(X, np.random.random(X.shape[1]))
    return X, y
