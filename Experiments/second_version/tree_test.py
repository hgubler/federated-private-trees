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
                    save_budget=False,
                    party_idx = [],
                    mixture_model = False,
                    n_centers = 0,
                    seed=0):
    dp_accuracies = []
    no_dp_accuracies = []
    local_accuracies = []
    normal_dt_accuracies = []
    dp_saving_accuracies = []
    for i in range(privacy_budgeds.shape[0]):
        
        
        accuracy = 0
        no_dp_accuracy = 0
        local_accuracy = 0
        normal_dt_accuracy = 0
        dp_saving_accuracy = 0
        for j in range(n_rep):
            no_dp_model = FederatedPrivateDecisionTree(max_depth=max_depth, 
                                                   min_samples_leaf=min_samples_leaf, 
                                                   diff_privacy=False, 
                                                   num_bins=num_bins, 
                                                   use_quantiles=use_quantiles)
            local_model = DecisionTree(max_depth=max_depth,
                                               min_samples_leaf=min_samples_leaf)
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
                                             save_budget=True,
                                             )
            simulate_data = SimulateData(n_features, mean_1, mean_2, rho_1, rho_2, 2 * n_samples, mixture_model=mixture_model, n_centers=n_centers, seed=seed)
            X, y = simulate_data.generate_data()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
            X_party_1 = X_train[party_idx[0]]
            y_party_1 = y_train[party_idx[0]]
            # fit models
            model.fit(X_train, y_train)
            no_dp_model.fit(X_train, y_train)
            local_model.fit(X_party_1, y_party_1)
            normal_dt_model.fit(X_train, y_train)
            dp_saving_model.fit(X_train, y_train)
            # make predictions
            predictions = model.predict(X_test)
            no_dp_predictions = no_dp_model.predict(X_test)
            local_predictions = local_model.predict(X_test)
            normal_dt_predictions = normal_dt_model.predict(X_test)
            dp_saving_predictions = dp_saving_model.predict(X_test)
            # calculate accuracies
            accuracy = accuracy + accuracy_score(y_test, predictions)
            no_dp_accuracy = no_dp_accuracy + accuracy_score(y_test, no_dp_predictions)
            local_accuracy = local_accuracy + accuracy_score(y_test, local_predictions)
            normal_dt_accuracy = normal_dt_accuracy + accuracy_score(y_test, normal_dt_predictions)
            dp_saving_accuracy = dp_saving_accuracy + accuracy_score(y_test, dp_saving_predictions)
            seed = seed + 1
        accuracy = accuracy / n_rep
        no_dp_accuracy = no_dp_accuracy / n_rep
        local_accuracy = local_accuracy / n_rep
        dp_saving_accuracy = dp_saving_accuracy / n_rep
        normal_dt_accuracy = normal_dt_accuracy / n_rep
        dp_accuracies.append(accuracy)
        no_dp_accuracies.append(no_dp_accuracy)
        local_accuracies.append(local_accuracy)
        normal_dt_accuracies.append(normal_dt_accuracy)
        dp_saving_accuracies.append(dp_saving_accuracy)
    return dp_accuracies, no_dp_accuracies, local_accuracies, normal_dt_accuracies, dp_saving_accuracies



n_samples = 100
privacy_budgeds = np.linspace(0.1, 10, 10)
rho_1 = 0.9
rho_2 = 0.9
min_samples_leaf = 10
num_bins = 10
use_quantiles = False
save_budget = False
mixture_model = True
n_centers = 5
seed = 0
party_idx = []
n_parties = 5
sample_per_party = n_samples / n_parties
if n_samples % n_parties != 0:
    raise ValueError('n_samples must be divisible by n_parties')
for i in range(n_parties):
    party_idx.append(np.arange(i * sample_per_party, (i + 1) * sample_per_party, dtype=int))
# simulation parameters
n_rep = 50
n_features_list = [10, 25, 50]
max_depth_list = [3, 7, 10]
fig, ax = plt.subplots(nrows=len(n_features_list), ncols=len(max_depth_list), sharex=True, sharey=True, figsize=(11, 8))

# Create lines for the legend
line1, = ax[0, 0].plot([], [], '-', label='differential privacy', color='red')
line2, = ax[0, 0].plot([], [], '-', label='no differential privacy mode', color='blue')
line3, = ax[0, 0].plot([], [], '-', label='local decision tree', color='green')
line4, = ax[0, 0].plot([], [], '-', label='differential privacy with saving budget', color='orange')
line5, = ax[0, 0].plot([], [], '-', label='centralized decision tree', color='black')

for n_features in n_features_list:
    for max_depth in max_depth_list:
        mean_1 = np.zeros(n_features)
        mean_2 = np.ones(n_features)
        accuracy, no_dp_accuracy, local_accuracy, normal_dt_accuracy, dp_saving_accuracy = accuracies_calc(privacy_budgeds=privacy_budgeds,
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
                                                                     save_budget=save_budget,
                                                                     party_idx=party_idx,
                                                                     mixture_model=mixture_model,
                                                                     n_centers=n_centers,
                                                                     seed=seed)
        ax[n_features_list.index(n_features), max_depth_list.index(max_depth)].plot(privacy_budgeds, accuracy, '-', color='red')
        ax[n_features_list.index(n_features), max_depth_list.index(max_depth)].plot(privacy_budgeds, no_dp_accuracy, '-', color='blue')
        ax[n_features_list.index(n_features), max_depth_list.index(max_depth)].plot(privacy_budgeds, local_accuracy, '-', color='green')
        ax[n_features_list.index(n_features), max_depth_list.index(max_depth)].plot(privacy_budgeds, dp_saving_accuracy, '-', color='orange')
        ax[n_features_list.index(n_features), max_depth_list.index(max_depth)].plot(privacy_budgeds, normal_dt_accuracy, '-', color='black')
        ax[n_features_list.index(n_features), max_depth_list.index(max_depth)].set_title('n_features = {}, max_depth = {}'.format(n_features, max_depth))
        if (n_features == n_features_list[-1]):
            ax[n_features_list.index(n_features), max_depth_list.index(max_depth)].set_xlabel('privacy budged')
        if (max_depth == max_depth_list[0]):
            ax[n_features_list.index(n_features), max_depth_list.index(max_depth)].set_ylabel('accuracy')
        print((1 + n_features_list.index(n_features)) * (1 + max_depth_list.index(max_depth)))
# set a general title for the figure specifying number of bins and number of samples
fig.suptitle('Accuracy vs. privacy budged (number of bins = {}, number of samples = {})'.format(num_bins, n_samples))
fig.legend(handles=[line1, line2, line3, line4, line5], loc='upper center', bbox_to_anchor=(0.515, 0.14))
fig.subplots_adjust(bottom=0.2) 
plt.savefig('Experiments/second_version/plots/accuracy_vs_privacy_budged_100_samples_gini.pdf', bbox_inches='tight')
plt.show()