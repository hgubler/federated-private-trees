import sys
sys.path.append('') # Adds higher directory to python modules path
from algorithms.federated_private_decision_tree_v2 import FederatedPrivateDecisionTree
from algorithms.decision_tree import DecisionTree
from Experiments.simulation.simulate_data import SimulateData
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_breast_cancer

def accuracies_calc(privacy_budgeds, 
                    X,
                    y,
                    max_depth=4, 
                    num_bins=10,
                    min_samples_leaf=10,
                    use_quantiles=False,
                    n_rep=100,
                    save_budget=False,
                    party_idx = [],
                    seed=0):
    dp_accuracies = []
    no_dp_accuracies = []
    sklearn_accuracies = []
    normal_dt_accuracies = []
    dp_saving_accuracies = []
    for i in range(privacy_budgeds.shape[0]):
        accuracy = 0
        no_dp_accuracy = 0
        sklearn_accuracy = 0
        normal_dt_accuracy = 0
        dp_saving_accuracy = 0
        for j in range(n_rep):
            no_dp_model = FederatedPrivateDecisionTree(max_depth=max_depth, 
                                                   min_samples_leaf=min_samples_leaf, 
                                                   diff_privacy=False, 
                                                   num_bins=num_bins, 
                                                   use_quantiles=use_quantiles)
            sklearn_model = DecisionTreeClassifier(max_depth=max_depth,
                                               min_samples_leaf=min_samples_leaf,
                                               criterion='entropy')
            normal_dt_model = DecisionTree(max_depth=max_depth,
                                               min_samples_leaf=min_samples_leaf)
            model = FederatedPrivateDecisionTree(max_depth=max_depth, 
                                             min_samples_leaf=min_samples_leaf, 
                                             privacy_budget=privacy_budgeds[i], 
                                             diff_privacy=True, num_bins=num_bins, 
                                             use_quantiles=use_quantiles,
                                             party_idx=party_idx,
                                             save_budget=save_budget)
            dp_saving_model = FederatedPrivateDecisionTree(max_depth=max_depth, 
                                             min_samples_leaf=min_samples_leaf, 
                                             privacy_budget=privacy_budgeds[i], 
                                             diff_privacy=True, num_bins=num_bins, 
                                             use_quantiles=use_quantiles,
                                             party_idx=party_idx,
                                             save_budget=True
                                             )
            # simulate data
            simulate = SimulateData(n_samples = 150, rho_1=0.9, rho_2=0.9)
            X, y = simulate.generate_data()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
            # fit models
            model.fit(X_train, y_train)
            no_dp_model.fit(X_train, y_train)
            sklearn_model.fit(X_train, y_train)
            normal_dt_model.fit(X_train, y_train)
            dp_saving_model.fit(X_train, y_train)
            # make predictions
            predictions = model.predict(X_test)
            no_dp_predictions = no_dp_model.predict(X_test)
            sklearn_predictions = sklearn_model.predict(X_test)
            normal_dt_predictions = normal_dt_model.predict(X_test)
            dp_saving_predictions = dp_saving_model.predict(X_test)
            # calculate accuracies
            accuracy = accuracy + accuracy_score(y_test, predictions)
            no_dp_accuracy = no_dp_accuracy + accuracy_score(y_test, no_dp_predictions)
            sklearn_accuracy = sklearn_accuracy + accuracy_score(y_test, sklearn_predictions)
            normal_dt_accuracy = normal_dt_accuracy + accuracy_score(y_test, normal_dt_predictions)
            dp_saving_accuracy = dp_saving_accuracy + accuracy_score(y_test, dp_saving_predictions)
            seed = seed + 1
        accuracy = accuracy / n_rep
        no_dp_accuracy = no_dp_accuracy / n_rep
        sklearn_accuracy = sklearn_accuracy / n_rep
        dp_saving_accuracy = dp_saving_accuracy / n_rep
        normal_dt_accuracy = normal_dt_accuracy / n_rep
        dp_accuracies.append(accuracy)
        no_dp_accuracies.append(no_dp_accuracy)
        sklearn_accuracies.append(sklearn_accuracy)
        normal_dt_accuracies.append(normal_dt_accuracy)
        dp_saving_accuracies.append(dp_saving_accuracy)
    return dp_accuracies, no_dp_accuracies, sklearn_accuracies, normal_dt_accuracies, dp_saving_accuracies


from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target
y[y==2] = 1
n_samples = X.shape[0] / 2
# split data into 5 parties, where in the last one we just take the leftover if n_samples is not divisible by 5
n_parties = 3
party_idx = []
sample_per_party = n_samples / n_parties
# adjust last sample per party if n_samples is not divisible by n_parties
if n_samples % n_parties != 0:
    sample_per_party = sample_per_party + n_samples % n_parties
for i in range(n_parties):
    party_idx.append(np.arange(i * sample_per_party, (i + 1) * sample_per_party, dtype=int))
privacy_budgeds = np.linspace(0.1, 10, 10)
n_rep = 50
dp_accuracies, no_dp_accuracies, sklearn_accuracies, normal_dt_accuracies, dp_saving_accuracies = accuracies_calc(privacy_budgeds, 
                                                                                                                 X,
                                                                                                                 y,
                                                                                                                 max_depth=10, 
                                                                                                                 num_bins=10,
                                                                                                                 min_samples_leaf=10,
                                                                                                                 use_quantiles=False,
                                                                                                                 n_rep=n_rep,
                                                                                                                 save_budget=False,
                                                                                                                 party_idx = party_idx,
                                                                                                                 seed=0)

#plot results
plt.plot(privacy_budgeds, dp_accuracies, label='DP', color='red')
plt.plot(privacy_budgeds, no_dp_accuracies, label='No DP', color='blue')
plt.plot(privacy_budgeds, sklearn_accuracies, label='Sklearn', color='green')
plt.plot(privacy_budgeds, normal_dt_accuracies, label='Normal DT', color='black')
plt.plot(privacy_budgeds, dp_saving_accuracies, label='DP with saving', color='orange')
plt.xlabel('Privacy budget')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
