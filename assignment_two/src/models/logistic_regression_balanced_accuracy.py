"""
This file contains code for running logistic regression on the CreditCardData-set. We use the built in LogisticRegression
model from sklearn, and perform a hyper-parameter optimization using grid search. We explore both l1 and l2 regularization.
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, cohen_kappa_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV

from assignment_two.src.utils.data_loaders import CreditCardData

credit_card_data = CreditCardData()
print(credit_card_data.num_features)
X = credit_card_data.X.numpy()
y = credit_card_data.y.numpy()

print(X.shape)
print(1 - sum(y) / len(y))
model_accuracy = []

ol_params = [
    {'penalty': ['none'],
     'solver': ['lbfgs']
     }
]

l1_params = [
    {
        'penalty': ['l1'],
        'C': np.logspace(-10, 10, 40),
        'solver': ['liblinear']
    },
]
l2_params = [
    {
        'penalty': ['l2'],
        'C': np.logspace(-10, 10, 40),
        'solver': ['liblinear']
    }
]

scorer = make_scorer(balanced_accuracy_score)

clf_ol = GridSearchCV(LogisticRegression(), ol_params, cv=5, verbose=True, n_jobs=-1, scoring=scorer)
clf_l1 = GridSearchCV(LogisticRegression(), l1_params, cv=5, verbose=True, n_jobs=-1, scoring=scorer)
clf_l2 = GridSearchCV(LogisticRegression(), l2_params, cv=5, verbose=True, n_jobs=-1, scoring=scorer)

clf_ol.fit(X, y)
clf_l1.fit(X, y)
clf_l2.fit(X, y)

l1_l = clf_l1.cv_results_['mean_test_score'] - clf_l1.cv_results_['std_test_score']
l1_h = clf_l1.cv_results_['mean_test_score'] + clf_l1.cv_results_['std_test_score']
l2_l = clf_l2.cv_results_['mean_test_score'] - clf_l2.cv_results_['std_test_score']
l2_h = clf_l2.cv_results_['mean_test_score'] + clf_l2.cv_results_['std_test_score']

clf_best = clf_l1.best_estimator_
clf_best.fit(X, y)

plt.semilogx(l1_params[0]['C'], clf_l1.cv_results_['mean_test_score'], label='l1', color='blue')
plt.semilogx(l2_params[0]['C'], clf_l2.cv_results_['mean_test_score'], label='l2', color='green')
# plt.fill_between(l1_l, l1_h, alpha=0.3, color='blue')
# plt.fill_between(l2_l, l2_h, alpha=0.3, color='green')

plt.xlabel('Regularization strength $\lambda$')
plt.ylabel('Balanced accuracy score')
plt.scatter(clf_l1.best_params_['C'], clf_l1.best_score_, label=rf'$\lambda_1 = {clf_l1.best_params_["C"]:.3e}$')
plt.scatter(clf_l2.best_params_['C'], clf_l2.best_score_, label=rf'$\lambda_2 = {clf_l2.best_params_["C"]:.3e}$')
plt.legend()

print(f'Best L0 = {clf_ol.best_score_}')
print(f'Best L1 = {clf_l1.best_score_}')
print(f'Best L2 = {clf_l2.best_score_}')

plt.savefig('credit_card_classification_balanced_accuracy.pdf')
plt.show()
