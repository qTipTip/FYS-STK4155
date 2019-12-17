import numpy as np
import sklearn
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNet
from skorch.dataset import CVSplit
from torch.utils.data import DataLoader

from assignment_two.src.models.neural_network import GenericNN
from assignment_two.src.utils.data_loaders import FrankeDataSet

franke_data_train = FrankeDataSet(num_points=2000)
train = DataLoader(franke_data_train)
X, y = franke_data_train.xy, franke_data_train.z_noise
print(X.shape, y.shape)
net = NeuralNet(
    GenericNN,
    module__num_input_features=2,
    module__num_output_features=1,
    module__num_hidden_layers=3,
    module__num_hidden_features=20,
    module__activation=torch.nn.functional.relu,
    criterion=torch.nn.MSELoss,
    optimizer=torch.optim.SGD,
    max_epochs=100,
    optimizer__nesterov=True,
    optimizer__momentum=0.9,
    # batch_size=20,
    train_split=CVSplit(cv=5),
)

params = {
    'lr': np.logspace(-5, 0, 20, endpoint=False),
}

reg = GridSearchCV(estimator=net, param_grid=params, scoring='neg_mean_squared_error', n_jobs=12)
reg.fit(X, y)

print(reg.cv_results_.keys())
mean_test = -reg.cv_results_['mean_test_score']
std_test = -reg.cv_results_['std_test_score']
plt.semilogx(params['lr'], mean_test, label='nn_validation', color='green')
plt.fill_between(params['lr'], mean_test - std_test,
                 mean_test + std_test, alpha=0.1)
# plt.semilogx(params['lr'], clf.cv_results_['mean_train_score'], label='nn_train', color='green')
plt.scatter(reg.best_params_['lr'], -reg.best_score_)
plt.title(rf'Optimal learning rate $\lambda = {reg.best_params_["lr"] :.04e}$')
plt.xlabel(rf'Learning rate $\lambda$')
plt.ylabel(rf'MSE')
plt.legend()
plt.savefig('regression_nn.pdf')
plt.show()
print(f'Best score = {-reg.best_score_}')
np.save('saved_scores/regression_nn_no_noise.npy', reg.cv_results_)

best_model = reg.best_estimator_
z_hat = best_model.predict(franke_data_train.xy)
best_score = r2_score(franke_data_train.z, z_hat)
print('R2', best_score)
