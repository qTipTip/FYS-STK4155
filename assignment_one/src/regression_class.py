import numpy as np
import sklearn.linear_model


class Regression(object):

    def __init__(self, X, y, inversion_method='direct'):
        self.X = X
        self.y = y
        self.xtx = np.dot(X.T, X)
        self.inv = np.linalg.inv if inversion_method == 'direct' else None

    def fit(self):
        raise NotImplementedError()

    def predict(self):
        self.y_hat = self.X.dot(self.beta)
        return self.y_hat

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


class OLS(Regression):
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

    def fit(self):
        pass

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

    def __init__(self, X, y, inversion_method='direct', lmbd=0.1):
        super().__init__(X, y, inversion_method)
        self.lmbd = lmbd
        self._regressor = sklearn.linear_model.Lasso(alpha=self.lmbd, fit_intercept=False)
        self._regressor.fit(self.X, self.y)

    @property
    def beta(self):
        return self._regressor.coef_


if __name__ == '__main__':
    x = np.random.randint(0, 10, (10, 6))
    y = np.random.randint(0, 10, (10,))

    O = OLS(x, y)
    R = Ridge(x, y, lmbd=0.1)
    L = Lasso(x, y, lmbd=0.1)
    print(O.beta)
    print(R.beta)
    print(L.beta)

    print(O.predict())
    for e in O, R, L:
        print(e.y_variance_estimate)
        print(e.beta_variance_estimate)
