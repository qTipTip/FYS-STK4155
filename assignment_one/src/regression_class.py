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
        return self.X.dot(self.beta)

    @property
    def beta(self):
        raise NotImplementedError()


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
    R = Ridge(x, y, lmbd=0)
    L = Lasso(x, y, lmbd=0)
    print(O.beta)
    print(R.beta)
    print(L.beta)

    print(O.predict())