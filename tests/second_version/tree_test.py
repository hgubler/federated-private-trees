import sys
sys.path.append('') # Adds higher directory to python modules path
from algorithms.federated_private_decision_tree_v2 import FederatedPrivateDecisionTree
from tests.simulation.simulate_data import SimulateData
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def accuracies_calc(privacy_budgeds, 
                    mean_1,
                    mean_2,
                    max_depth=4, 
                    num_bins=10,
                    min_samples_leaf=10,
                    use_quantiles=False,
                    n_rep=100,
                    rho_1=0.5,
                    rho_2=0.9,
                    n_samples=1000,
                    n_features=10,
                    seed=0):
    dp_accuracies = []
    no_dp_accuracies = []
    sklearn_accuracies = []
    for i in range(privacy_budgeds.shape[0]):
        model = FederatedPrivateDecisionTree(max_depth=max_depth, 
                                             min_samples_leaf=min_samples_leaf, 
                                             privacy_budget=privacy_budgeds[i], 
                                             diff_privacy=True, num_bins=num_bins, 
                                             use_quantiles=use_quantiles)
        no_dp_model = FederatedPrivateDecisionTree(max_depth=max_depth, 
                                                   min_samples_leaf=min_samples_leaf, 
                                                   diff_privacy=False, 
                                                   num_bins=num_bins, 
                                                   use_quantiles=use_quantiles)
        sklearn_model = DecisionTreeClassifier(max_depth=max_depth,
                                               min_samples_leaf=min_samples_leaf,
                                               criterion='entropy')
        accuracy = 0
        no_dp_accuracy = 0
        sklearn_accuracy = 0
        for j in range(n_rep):
            simulate_data = SimulateData(n_features, mean_1, mean_2, rho_1, rho_2, n_samples, seed)
            X, y = simulate_data.generate_data()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
            # fit models
            model.fit(X_train, y_train)
            no_dp_model.fit(X_train, y_train)
            sklearn_model.fit(X_train, y_train)
            # make predictions
            predictions = model.predict(X_test)
            no_dp_predictions = no_dp_model.predict(X_test)
            sklearn_predictions = sklearn_model.predict(X_test)
            # calculate accuracies
            accuracy = accuracy + accuracy_score(y_test, predictions)
            no_dp_accuracy = no_dp_accuracy + accuracy_score(y_test, no_dp_predictions)
            sklearn_accuracy = sklearn_accuracy + accuracy_score(y_test, sklearn_predictions)
            seed = seed + 1
        accuracy = accuracy / n_rep
        no_dp_accuracy = no_dp_accuracy / n_rep
        sklearn_accuracy = sklearn_accuracy / n_rep
        dp_accuracies.append(accuracy)
        no_dp_accuracies.append(no_dp_accuracy)
        sklearn_accuracies.append(sklearn_accuracy)
    return dp_accuracies, no_dp_accuracies, sklearn_accuracies



n_samples = 100
privacy_budgeds = np.linspace(0.1, 10, 10)
rho_1 = 0.5
rho_2 = 0.9
min_samples_leaf = 10
num_bins = 10
use_quantiles = False
seed = 0
# simulation parameters
n_rep = 100
n_features_list = [10, 25, 50]
max_depth_list = [4, 7, 10]
fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True)

# Create lines for the legend
line1, = ax[0, 0].plot([], [], '-', label='differential privacy', color='red')
line2, = ax[0, 0].plot([], [], '-', label='no differential privacy mode', color='blue')
line3, = ax[0, 0].plot([], [], '-', label='normal sklearn decision tree', color='green')

for n_features in n_features_list:
    for max_depth in max_depth_list:
        mean_1 = np.zeros(n_features)
        mean_2 = np.ones(n_features)
        accuracy, no_dp_accuracy, sklearn_accuracy = accuracies_calc(privacy_budgeds=privacy_budgeds,
                                                                     max_depth=max_depth,
                                                                     num_bins=num_bins,
                                                                     min_samples_leaf=min_samples_leaf,
                                                                     use_quantiles=use_quantiles,
                                                                     n_rep=n_rep,
                                                                     rho_1=rho_1,
                                                                     rho_2=rho_2,
                                                                     mean_1 = mean_1,
                                                                     mean_2 = mean_2,
                                                                     n_samples=n_samples,
                                                                     n_features=n_features,
                                                                     seed=seed)
        ax[n_features_list.index(n_features), max_depth_list.index(max_depth)].plot(privacy_budgeds, accuracy, '-', color='red')
        ax[n_features_list.index(n_features), max_depth_list.index(max_depth)].plot(privacy_budgeds, no_dp_accuracy, '-', color='blue')
        ax[n_features_list.index(n_features), max_depth_list.index(max_depth)].plot(privacy_budgeds, sklearn_accuracy, '-', color='green')
        ax[n_features_list.index(n_features), max_depth_list.index(max_depth)].set_xlabel('privacy budged')
        ax[n_features_list.index(n_features), max_depth_list.index(max_depth)].set_title('n_features = {}, max_depth = {}'.format(n_features, max_depth))
        print((1 + n_features_list.index(n_features)) * (1 + max_depth_list.index(max_depth)))
# set a general title for the figure specifying number of bins and number of samples
fig.suptitle('Accuracy vs. privacy budged (number of bins = {}, number of samples = {})'.format(num_bins, n_samples))
fig.legend(handles=[line1, line2, line3], loc='upper right')
plt.savefig('tests/second_version/plots/accuracy_vs_privacy_budged_100_samples_gini.png')
plt.show()