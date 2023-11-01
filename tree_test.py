from federated_private_decision_tree import FederatedPrivateDecisionTree
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import numpy as np
# import breast cancer dataset from sklearn
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target
# create train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


# create and fit model with exponential leaf mechanism
exp_model = FederatedPrivateDecisionTree(max_depth=10, privacy_budget=1, diff_privacy=True, leaf_mechanism='exponential')
exp_model.fit(X_train, y_train)
# make predictions
predictions = exp_model.predict(X_test)
# calculate accuracy
exp_accuracy = accuracy_score(y_test, predictions)
print("Accuracy with exponential leaf labelling: " + str(exp_accuracy))

# create and fit model with laplace leaf mechanism
laplace_model = FederatedPrivateDecisionTree(max_depth=10, privacy_budget=1, diff_privacy=True, leaf_mechanism='laplace')
laplace_model.fit(X_train, y_train)
# make predictions
predictions = laplace_model.predict(X_test)
# calculate accuracy
laplace_accuracy = accuracy_score(y_test, predictions)
print("Accuracy with laplace leaf labelling: " + str(laplace_accuracy))

# create and fit model without differential privacy
no_dp_model = FederatedPrivateDecisionTree(max_depth=10, privacy_budget=1, diff_privacy=False)
no_dp_model.fit(X_train, y_train)
# make predictions
predictions = no_dp_model.predict(X_test)
# calculate accuracy
no_dp_accuracy = accuracy_score(y_test, predictions)
print("Accuracy without differential privacy: " + str(no_dp_accuracy))


# plot accuracies of all three models vs depth on x-axis
depths = [i for i in range(1, 20)]
exp_accuracies = []
laplace_accuracies = []
no_dp_accuracies = []
for depth in depths:
    exp_model = FederatedPrivateDecisionTree(max_depth=depth, privacy_budget=1, diff_privacy=True, leaf_mechanism='exponential')
    exp_model.fit(X_train, y_train)
    predictions = exp_model.predict(X_test)
    exp_accuracy = accuracy_score(y_test, predictions)
    exp_accuracies.append(exp_accuracy)

    laplace_model = FederatedPrivateDecisionTree(max_depth=depth, privacy_budget=1, diff_privacy=True, leaf_mechanism='laplace')
    laplace_model.fit(X_train, y_train)
    predictions = laplace_model.predict(X_test)
    laplace_accuracy = accuracy_score(y_test, predictions)
    laplace_accuracies.append(laplace_accuracy)

    no_dp_model = FederatedPrivateDecisionTree(max_depth=depth, privacy_budget=1, diff_privacy=False)
    no_dp_model.fit(X_train, y_train)
    predictions = no_dp_model.predict(X_test)
    no_dp_accuracy = accuracy_score(y_test, predictions)
    no_dp_accuracies.append(no_dp_accuracy)


plt.plot(depths, exp_accuracies, '-', label='exponential')
plt.plot(depths, laplace_accuracies, '-', label='laplace')
plt.plot(depths, no_dp_accuracies, '-', label='no differential privacy')
plt.title("Accuracy vs depth")
plt.xlabel("depth")
plt.ylabel("accuracy")
plt.legend()
#plt.show()
# safe plot in plots folder
plt.savefig("plots/accuracy_vs_depth.png")


# repeat the same but now take privacy budged on x-axis
privacy_budgets = np.linspace(0.1, 2, 10)
exp_accuracies = []
laplace_accuracies = []
no_dp_accuracies = []
for privacy_budget in privacy_budgets:
    exp_model = FederatedPrivateDecisionTree(max_depth=10, privacy_budget=privacy_budget, diff_privacy=True, leaf_mechanism='exponential')
    exp_model.fit(X_train, y_train)
    predictions = exp_model.predict(X_test)
    exp_accuracy = accuracy_score(y_test, predictions)
    exp_accuracies.append(exp_accuracy)

    laplace_model = FederatedPrivateDecisionTree(max_depth=10, privacy_budget=privacy_budget, diff_privacy=True, leaf_mechanism='laplace')
    laplace_model.fit(X_train, y_train)
    predictions = laplace_model.predict(X_test)
    laplace_accuracy = accuracy_score(y_test, predictions)
    laplace_accuracies.append(laplace_accuracy)

    no_dp_model = FederatedPrivateDecisionTree(max_depth=10, privacy_budget=privacy_budget, diff_privacy=False)
    no_dp_model.fit(X_train, y_train)
    predictions = no_dp_model.predict(X_test)
    no_dp_accuracy = accuracy_score(y_test, predictions)
    no_dp_accuracies.append(no_dp_accuracy)

plt.figure()
plt.plot(privacy_budgets, exp_accuracies, '-', label='exponential')
plt.plot(privacy_budgets, laplace_accuracies, '-', label='laplace')
plt.plot(privacy_budgets, no_dp_accuracies, '-', label='no differential privacy')
plt.title("Accuracy vs privacy budget")
plt.xlabel("privacy budget")
plt.ylabel("accuracy")
plt.legend()
#plt.show()
# safe plot in plots folder
plt.savefig("plots/accuracy_vs_privacy_budget.png")
print("hi")







