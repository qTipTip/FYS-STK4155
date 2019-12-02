import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from assignment_two.src.utils.data_loaders import CreditCardData

credit_card_data = CreditCardData()
X = credit_card_data.X.numpy()
y = credit_card_data.y.numpy()

model_accuracy = []

l1_params = [
    {
        'penalty': ['l1'],
        'C': np.logspace(-4, 4, 40),
        'solver': ['liblinear']
    },
]
l2_params = [
    {
        'penalty': ['l2'],
        'C': np.logspace(-4, 4, 40),
        'solver': ['liblinear']
    }
]

logspace = np.logspace(0, -3, 100)
clf_l1 = GridSearchCV(LogisticRegression(), l1_params, cv=5, verbose=True, n_jobs=-1)
clf_l2 = GridSearchCV(LogisticRegression(), l2_params, cv=5, verbose=True, n_jobs=-1)

clf_l1.fit(X, y)
clf_l2.fit(X, y)

plt.semilogx(l1_params[0]['C'], clf_l1.cv_results_['mean_test_score'], label='l1')
plt.semilogx(l2_params[0]['C'], clf_l2.cv_results_['mean_test_score'], label='l2')

plt.legend()
plt.show()
