import numpy as np
from sklearn.model_selection import GridSearchCV
import torch
import matplotlib.pyplot as plt

from skorch.dataset import CVSplit

from assignment_two.src.models.neural_network import GenericNN, RegularizedNeuralNetClassifier
from assignment_two.src.utils.data_loaders import CreditCardData

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
np.save('saved_scores/logistic_regression_nn.npy', clf.cv_results_)
