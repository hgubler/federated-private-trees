import sys
sys.path.append('') # Adds higher directory to python modules path
from algorithms.federated_private_decision_tree_v2 import FederatedPrivateDecisionTree
from Experiments.simulation.simulate_data import SimulateData
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

privacy_budget = 5
n_props = 50
leaf_props = np.linspace(1 / n_props, 1, n_props, endpoint=False)
bounds_props = np.linspace(1 / n_props, 1, n_props, endpoint=False)
n_rep = 10
n_samples = 300
n_features = 10
n_centers = 5
mixture_model = True
seed = 0
rho_1 = 0.3
rho_2 = 0.3
# party indexing
party_idx = []
n_parties = 5
sample_per_party = n_samples / n_parties
if n_samples % n_parties != 0:
    raise ValueError('n_samples must be divisible by n_parties')
for i in range(n_parties):
    party_idx.append(np.arange(i * sample_per_party, (i + 1) * sample_per_party, dtype=int))


def no_budged_saving_accuracy(leaf_prop, n_rep, n_samples, n_features, n_centers, mixture_model, rho_1, rho_2, privacy_budget, seed):
    no_budget_saving_accuracy = 0
    for i in range(n_rep):
        random_state = seed
        simulate_data = SimulateData(n_features=n_features, rho_1=rho_1, rho_2=rho_2, n_samples=2 * n_samples, mixture_model=mixture_model, n_centers=n_centers, seed=seed)
        X, y = simulate_data.generate_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)
        no_budged_saving_model = FederatedPrivateDecisionTree(max_depth = 7,
                                                              privacy_budget = privacy_budget,
                                                              leaf_privacy_proportion=leaf_prop,
                                                              save_budget=False,
                                                              random_state=seed)
        no_budged_saving_model.fit(X_train, y_train)
        y_pred = no_budged_saving_model.predict(X_test)
        no_budged_saving_accuracy = accuracy_score(y_test, y_pred)
        no_budget_saving_accuracy += no_budged_saving_accuracy
        seed += 1
    no_budget_saving_accuracy = no_budget_saving_accuracy / n_rep
    return no_budget_saving_accuracy

def budget_saving_accuracy(leaf_prop, bounds_prop, n_rep, n_samples, n_features, n_centers, mixture_model, rho_1, rho_2, privacy_budget, seed):
    budget_saving_accuracy = 0
    for i in range(n_rep):
        random_state = seed
        simulate_data = SimulateData(n_features=n_features, rho_1=rho_1, rho_2=rho_2, n_samples=2 * n_samples, mixture_model=mixture_model, n_centers=n_centers, seed=seed)
        X, y = simulate_data.generate_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)
        budged_saving_model = FederatedPrivateDecisionTree(max_depth = 7,
                                                           privacy_budget = privacy_budget,
                                                           leaf_privacy_proportion=leaf_prop,
                                                           bounds_privacy_proportion=bounds_prop,
                                                           save_budget=True,
                                                           random_state=seed)
        budged_saving_model.fit(X_train, y_train)
        y_pred = budged_saving_model.predict(X_test)
        budged_saving_accuracy = accuracy_score(y_test, y_pred)
        budget_saving_accuracy += budged_saving_accuracy
        seed += 1
    budget_saving_accuracy = budget_saving_accuracy / n_rep
    return budget_saving_accuracy


budged_saving_accuracies = np.zeros((len(leaf_props), len(bounds_props)))
no_budged_saving_accuracies = np.zeros(len(leaf_props))

for i, leaf_prop in enumerate(leaf_props):
    # print the round we are currently in with the index of leaf_prop
    print('round: ', np.where(leaf_props == leaf_prop)[0][0])
    no_budged_saving_accuracies[i] = no_budged_saving_accuracy(leaf_prop, n_rep, n_samples, n_features, n_centers, mixture_model, rho_1, rho_2, privacy_budget, seed)
    for j, bounds_prop in enumerate(bounds_props):
        budged_saving_accuracies[j, i] = budget_saving_accuracy(leaf_prop, bounds_prop, n_rep, n_samples, n_features, n_centers, mixture_model, rho_1, rho_2, privacy_budget, seed)

fig1 = plt.figure(figsize=(6, 6))

# only take every 5-th value of leaf_prop and no_budget_saving_accuracies to plot for better visibility
# plot results first for no budget saving model
plt.plot(leaf_props[::5], no_budged_saving_accuracies[::5], '-o', color='red')
plt.xlabel('leaf privacy budget proportion')
plt.ylabel('accuracy')
plt.savefig('Experiments/main_report_plots/budget_distribution_no_saving.pdf')

fig2 = plt.figure(figsize=(6, 6))
# plot results for budget saving model as a heatmap for the matrix of leaf and bounds privacy proportions
im = plt.imshow(budged_saving_accuracies, cmap='hot', interpolation='nearest')
plt.colorbar(im)
plt.xlabel('leaf privacy budget proportion')
plt.ylabel('bounds privacy budget proportion')
plt.savefig('Experiments/main_report_plots/budged_distribution_saving.pdf')
plt.show()





