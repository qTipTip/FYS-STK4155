import numpy as np


class Regression():

    def __init__(self, X, y, inversion_method='direct'):
        self.X = X
        self.y = y
        self.XTX = np.dot(X.T, X)
        self.inv_method = np.linalg.inv if inversion_method == 'direct' else None

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self):
        """
        Returns the prediction corresponding to optimal model parameters
        :return: y = self.X.dot(self.beta)
        """

        return self.X.dot(self.beta)


class OLS(Regression):

    def fit(self):
        """
        Returns the analytically optimal model parameters for ordinary least squares.
        :return:
        """

        if hasattr(self, 'beta'):
            return self.beta
        else:
            xtx_inv = self.inv_method(self.XTX)
            self.beta = xtx_inv.dot(self.X.T).dot(self.y)
            return self.beta


class Ridge(Regression):

    def __init__(self, X, y, inversion_method='direct', lambd=1.0):
        super().__init__(X, y, inversion_method)
        self.lambd = lambd

    def fit(self):
        """
        Returns the analytically optimal model parameters for ordinary least squares.
        :return:
        """

        if hasattr(self, 'beta'):
            return self.beta
        else:
            xtx_inv = self.inv_method(self.XTX + self.lambd * np.eye(self.XTX.shape[0]))
            self.beta = xtx_inv.dot(self.X.T).dot(self.y)
            return self.beta


if __name__ == '__main__':
    x = np.linspace(-1, 5, 20)
    X = np.ones((20, 3))
    X[:, 1] = x
    X[:, 2] = x ** 2
    # X = sklearn.preprocessing.PolynomialFeatures(degree=2).fit_transform(x)
    print(X.shape)
    y = 3 * x ** 3 + x

    import matplotlib.pyplot as plt

    O = OLS(X, y)
    b = O.fit()
    y_hat = O.predict()
    plt.plot(x, y_hat, label='OLS')
    for l in [10 ** (- i - 1) for i in range(4)]:
        R = Ridge(X, y, lambd=l)
        b = R.fit()
        y_hat = R.predict()
        plt.plot(x, y_hat, label=f'l = {l}')

    print(b.shape)
    print(b)

    print(y)

    print(X.shape)
    plt.scatter(x, y)
    plt.legend()
    plt.show()
