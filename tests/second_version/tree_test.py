import sys
sys.path.append('') # Adds higher directory to python modules path
from algorithms.federated_private_decision_tree_v2 import FederatedPrivateDecisionTree
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

def accuracies_calc(privacy_budgeds, X_train, X_test, y_train, y_test, k, max_depth=4, num_bins=10):
    accuracies = []
    for i in range(privacy_budgeds.shape[0]):
        model = FederatedPrivateDecisionTree(max_depth=max_depth, privacy_budget=privacy_budgeds[i], diff_privacy=True, num_bins=num_bins)
        accuracy = 0
        for j in range(k):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy + accuracy_score(y_test, predictions)
        accuracies.append(accuracy / k)
    
    no_dp_model = FederatedPrivateDecisionTree(max_depth=max_depth, diff_privacy=False, num_bins=num_bins)
    no_dp_model.fit(X_train, y_train)
    no_dp_predictions = no_dp_model.predict(X_test)
    no_dp_accuracy = accuracy_score(y_test, no_dp_predictions)
    return accuracies, no_dp_accuracy


# experiment parameters
privacy_budgeds = np.linspace(0.1, 20, 20)
max_depth = 4
num_bins = 10
k = 10

# iris dataset experiment
iris = load_iris()
X = iris.data
y = iris.target
y[y==2] = 1
X_train, X_test, y_train, y_test = train_test_split(X, y)
accuracies, no_dp_accuracy = accuracies_calc(privacy_budgeds, X_train, X_test, y_train, y_test, k, max_depth=4, num_bins=num_bins)
sklearn_model = DecisionTreeClassifier(max_depth=max_depth)
sklearn_model.fit(X_train, y_train)
sklearn_predictions = sklearn_model.predict(X_test)
sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)
plt.figure()
plt.plot(privacy_budgeds, accuracies, '-', label='differential privacy')
plt.plot(privacy_budgeds, [no_dp_accuracy] * privacy_budgeds.shape[0], label = 'no differential privacy mode')
plt.plot(privacy_budgeds, [sklearn_accuracy] * privacy_budgeds.shape[0], label = 'normal sklearn decision tree')
plt.legend()
plt.xlabel('pruvacy budged')
plt.ylabel('accuracy')
plt.title('Accuracies Iris Dataset')
plt.suptitle('max_depth = ' +  str(max_depth) + ", bins = " + str(num_bins))
plt.savefig('tests/second_version/plots/accuracies_iris.png')

# breastcancer dataset experiment
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
accuracies, no_dp_accuracy = accuracies_calc(privacy_budgeds, X_train, X_test, y_train, y_test, k, max_depth=4, num_bins=num_bins)
sklearn_model = DecisionTreeClassifier(max_depth=max_depth)
sklearn_model.fit(X_train, y_train)
sklearn_predictions = sklearn_model.predict(X_test)
sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)
plt.figure()
plt.plot(privacy_budgeds, accuracies, '-', label='differential privacy')
plt.plot(privacy_budgeds, [no_dp_accuracy] * privacy_budgeds.shape[0], label = 'no differential privacy mode')
plt.plot(privacy_budgeds, [sklearn_accuracy] * privacy_budgeds.shape[0], label = 'normal sklearn decision tree')
plt.legend()
plt.xlabel('pruvacy budged')
plt.ylabel('accuracy')
plt.title('Accuracies Breastcancer Dataset')
plt.suptitle('max_depth = ' +  str(max_depth) + ", bins = " + str(num_bins))
plt.savefig('tests/second_version/plots/accuracies_breastcancer.png')





