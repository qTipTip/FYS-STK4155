import numpy as np
import sklearn.linear_model


class Regression(object):
    name = 'RegressionBaseClass'

    def __init__(self, X, y, inversion_method='direct'):
        self.X = X
        self.y = y
        self.xtx = np.dot(X.T, X)
        self.inv = np.linalg.inv if inversion_method == 'direct' else Regression.svd_inv

    def fit(self):
        raise NotImplementedError()

    def predict(self, X=None):
        if X is None:
            X = self.X
        y_hat = X.dot(self.beta)
        if X is None:
            self._y_hat = y_hat
        return y_hat

    @property
    def y_hat(self):
        if not hasattr(self, '_y_hat'):
            return self.predict()
        else:
            return self._y_hat

    @property
    def y_variance_estimate(self):
        """
        Return an unbiased estimate of the response variance (assuming all y_i are uncorrelated
        and constant variance sigma^2).

        :return: estimate of sigma^2
        """

        if not hasattr(self, 'y_hat'):
            self.predict()
        n, p = self.X.shape

        return 1 / (n - p - 1) * np.sum((self.y - self.y_hat) ** 2)

    @property
    def beta(self):
        raise NotImplementedError()

    @property
    def beta_covariance_estimate(self):
        """
        Returns the estimated covariance matrix for beta, which is a matrix of shape (n, n)

        :return: covariance matrix estimate
        """
        return self.inv(self.xtx) * self.y_variance_estimate

    @property
    def beta_variance_estimate(self):
        """
        Returns the estimated variances for beta, simply by extracting the diagonal from the covariance matrix.
        This is an array of shape (n,)

        :return: variance of beta
        """
        return np.diag(self.beta_covariance_estimate)

    @property
    def beta_std_dev_estimate(self):
        """
        Returns the estimated standard deviations for beta by taking the square root of the variance estimate.

        :return: standard deviation of beta
        """
        return np.sqrt(self.beta_variance_estimate)

    def mse(self):
        """
        Computes the mean squared error of the prediction y_hat to the ground truth y.

        :return: mean squared error
        """

        n = self.y.shape[0]
        return ((self.y - self.y_hat) ** 2).sum() / n

    def r2(self):
        """
        Computes the r2-score (or coefficient of determination) of the prediction y_hat to the ground truth y.

        :param y: ground truth
        :param y_hat: prediction
        :return: r2-score
        """

        y_mean = self.y.mean()

        num = ((self.y - self.y_hat) ** 2).sum()
        den = ((self.y - y_mean) ** 2).sum()

        return 1 - num / den

    @staticmethod
    def svd_inv(A):
        """
        Returns the inverse of the matrix A based on a singular value decomposition of A.

        :param A: matrix to invert
        :return: inv(A)
        """

        u, sigma, v_transposed = np.linalg.svd(A)
        m = u.shape[0]
        n = v_transposed.shape[0]

        D = np.zeros((m, n))
        for i in range(n):
            D[i, i] = sigma[i]

        return v_transposed.T @ (np.linalg.pinv(D) @ u.T)


class OLS(Regression):
    name = 'OLS'
    @property
    def beta(self):
        if hasattr(self, '_beta'):
            return self._beta
        else:
            self._beta = self.inv(self.xtx).dot(self.X.T).dot(self.y)
            return self._beta

    def fit(self):
        xtx_inv = self.inv(self.xtx)
        self._beta = xtx_inv.dot(self.X.T).dot(self.y)


class Ridge(Regression):
    name = 'Ridge'

    def __init__(self, X, y, inversion_method='direct', lmbd=0.1):
        super().__init__(X, y, inversion_method)
        self.lmbd = lmbd

    @property
    def beta(self):
        if hasattr(self, '_beta'):
            return self._beta
        else:
            self._beta = self.inv(self.xtx + self.lmbd * np.eye(self.xtx.shape[0])).dot(self.X.T).dot(self.y)
            return self._beta


class Lasso(Regression):
    name = 'Lasso'

    def __init__(self, X, y, inversion_method='direct', lmbd=0.1):
        super().__init__(X, y, inversion_method)
        self.lmbd = lmbd
        self._regressor = sklearn.linear_model.Lasso(alpha=self.lmbd, fit_intercept=False)
        self._regressor.fit(self.X, self.y)
        self.name = 'Lasso'

    @property
    def beta(self):
        return self._regressor.coef_


if __name__ == '__main__':
    x = np.linspace(0, 1, 10000)
    X = np.ones((10000, 2))
    X[:, 1] = x
    y = 4 * x ** 4 + np.random.normal(0, np.sqrt(9), 10000)

    O = OLS(X, y, inversion_method='svd')
    R = Ridge(X, y, lmbd=0.1)
    L = Lasso(X, y, lmbd=0.1)
