import numpy as np
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from skorch.dataset import CVSplit

from assignment_two.src.utils.data_loaders import CreditCardData


class GenericNN(nn.Module):

    def __init__(self, num_input_features=10, num_output_features=10, num_hidden_layers=1, num_hidden_features=20, activation=F.relu,
                 alpha=1, regularization='none'):
        super().__init__()

        self.input_layer = nn.Linear(num_input_features, num_hidden_features).double()
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(num_hidden_features, num_hidden_features).double() for i in range(num_hidden_layers)])
        self.output_layer = nn.Linear(num_hidden_features, num_output_features).double()
        self.f = activation

    def forward(self, x):
        x = self.f(self.input_layer(x))
        for l in self.hidden_layers:
            x = self.f(l(x))
        x = self.output_layer(x)
        return x


class RegularizedNeuralNetClassifier(NeuralNetClassifier):

    def __init__(self, module, alpha=1, regularizer='none', criterion=torch.nn.NLLLoss,
                 train_split=CVSplit(5, stratified=True),
                 classes=None,
                 *args,
                 **kwargs):
        self.alpha = alpha
        self.regularizer = regularizer

        if 'regularizer' in kwargs.keys():
            kwargs.pop('regularizer')
        if 'alpha' in kwargs.keys():
            kwargs.pop('alpha')

        super().__init__(module, *args, criterion=criterion, train_split=train_split, classes=classes, **kwargs)

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        loss = super().get_loss(y_pred, y_true, *args, **kwargs)

        if self.regularizer == 'none':
            pass
        elif self.regularizer == 'l1':
            loss += self.alpha * sum([w.abs().sum() for w in self.module_.parameters()])
        elif self.regularizer == 'l2':
            loss += self.alpha * sum([(w.abs() ** 2).sum() for w in self.module_.parameters()])
        return loss


credit_card_data = CreditCardData()

NN = GenericNN(num_input_features=credit_card_data.num_features, num_hidden_layers=3, num_output_features=2)

net = RegularizedNeuralNetClassifier(
    GenericNN,
    module__num_input_features=credit_card_data.num_features,
    module__num_hidden_layers=3,
    module__num_output_features=2,
    criterion=torch.nn.CrossEntropyLoss,
    train_split=CVSplit(cv=5, random_state=42),
    optimizer=torch.optim.SGD,
    max_epochs=20,
    optimizer__momentum=0.9,
    regularizer='l2',
    alpha=0
)

params = {
    'lr': np.logspace(-4, 0, 30),
    # 'module__num_hidden_layers' : range(0, 5),
    # 'module__hidden_layer_size' : [5, 10, 15, 20]
}
clf = GridSearchCV(
    net,
    params,
    n_jobs=12,
    cv=5
)
clf.fit(credit_card_data.X.numpy(), credit_card_data.y.numpy())

plt.semilogx(params['lr'], clf.cv_results_['mean_test_score'], label='nn_validation', color='green')
plt.fill_between(params['lr'], clf.cv_results_['mean_test_score'] - clf.cv_results_['std_test_score'],
                 clf.cv_results_['mean_test_score'] + clf.cv_results_['std_test_score'], alpha=0.1)
# plt.semilogx(params['lr'], clf.cv_results_['mean_train_score'], label='nn_train', color='green')
plt.scatter(clf.best_params_['lr'], clf.best_score_)
plt.title(rf'Optimal learning rate $\lambda = {clf.best_params_["lr"] :.04e}$')
plt.xlabel(rf'Learning rate $\lambda$')
plt.ylabel(rf'Classification accuracy')
plt.legend()
plt.savefig('classification_credit_cards_nn.pdf')
plt.show()
print(f'Best score = {clf.best_score_}')
np.save('logistic_regression_nn.npy', clf.cv_results_)
