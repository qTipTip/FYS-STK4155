import numpy as np
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

from assignment_two.src.utils.data_loaders import FrankeDataSet

franke_data = FrankeDataSet(num_points=2000, signal_to_noise=0.1, noisy=True, random_seed=5, degree=5)
print(franke_data.X.shape, franke_data.z_noise.shape)

regressors = [('Lasso', Lasso, 4.23e-2), ('Ridge', Ridge, 2.36e2), ('OLS', LinearRegression, 0)]

for name, reg_type, alpha in regressors:
    if name != 'OLS':
        reg = reg_type(alpha=alpha)
    else:
        reg = reg_type()
    reg.fit(franke_data.X, franke_data.z_noise)
    z_hat = reg.predict(franke_data.X)
    print(name, 'R2', reg.score(franke_data.X, franke_data.z))
    print(name, 'MSE', mean_squared_error(franke_data.z, z_hat))
