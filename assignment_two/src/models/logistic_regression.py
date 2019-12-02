import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from assignment_two.src.utils.data_loaders import CreditCardData

credit_card_data = CreditCardData()
X = credit_card_data.X.numpy()
y = credit_card_data.y.numpy()

print()
clf = LogisticRegression(penalty='l2', fit_intercept=False, solver='lbfgs')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf.fit(X_train, y_train)
print(f'scikit-learn logistic regression accuracy = {clf.score(X_test, y_test):.03f}')
