import sys
sys.path.append('') # Adds higher directory to python modules path
from algorithms.federated_private_decision_tree_v2 import FederatedPrivateDecisionTree
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
# convert y to 0 and 1
y = data.target
y[y == 2] = 1
# create train test split
X_train, X_test, y_train, y_test = train_test_split(X, y)
privacy_budgeds = np.linspace(1, 50, 100)
accuracies = []
for i in range(privacy_budgeds.shape[0]):
    model = FederatedPrivateDecisionTree(max_depth=4, privacy_budget=privacy_budgeds[i], diff_privacy=True, num_bins=10)
    accuracy = 0
    k = 20
    for j in range(k):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy + accuracy_score(y_test, predictions)
    accuracies.append(accuracy / k)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
plt.plot(privacy_budgeds, accuracies, '-', label='differential privacy')
plt.plot(privacy_budgeds, [accuracy] * privacy_budgeds.shape[0], label='no differential privacy')
plt.xlabel('privacy budget')
plt.ylabel('accuracy')
plt.legend()
plt.show()
