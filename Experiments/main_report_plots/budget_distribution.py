import sys
sys.path.append('') # Adds higher directory to python modules path
from experiments.evaluation_functions import budget_saving_accuracy, no_budged_saving_accuracy
from matplotlib import pyplot as plt
import numpy as np

# set parameters
privacy_budget = 5
n_props = 50
leaf_props = np.linspace(1 / n_props, 1, n_props, endpoint=False)
bounds_props = np.linspace(1 / n_props, 1, n_props, endpoint=False)
n_rep = 50
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



# compute the accuracies for the different leaf and bounds proportions
budged_saving_accuracies = np.zeros((len(leaf_props), len(bounds_props)))
no_budged_saving_accuracies = np.zeros(len(leaf_props))

for i, leaf_prop in enumerate(leaf_props):
    # print the round we are currently in with the index of leaf_prop
    print('round: ', np.where(leaf_props == leaf_prop)[0][0])
    no_budged_saving_accuracies[i] = no_budged_saving_accuracy(leaf_prop, n_rep, n_samples, n_features, n_centers, mixture_model, rho_1, rho_2, privacy_budget, seed)
    for j, bounds_prop in enumerate(bounds_props):
        budged_saving_accuracies[j, i] = budget_saving_accuracy(leaf_prop, bounds_prop, n_rep, n_samples, n_features, n_centers, mixture_model, rho_1, rho_2, privacy_budget, seed)

# plot results first for no budget saving model
fig1 = plt.figure(figsize=(6, 6))

# only take every 5-th value of leaf_prop and no_budget_saving_accuracies to plot for better visibility
# plot results first for no budget saving model
plt.plot(leaf_props[::5], no_budged_saving_accuracies[::5], '-o', color='red')
plt.xlabel('leaf privacy budget proportion')
plt.ylabel('test accuracy')
plt.savefig('Experiments/main_report_plots/budget_distribution_no_saving.pdf')

# plot results for budget saving model as a heatmap for the matrix of leaf and bounds privacy proportions
fig2 = plt.figure(figsize=(6, 6))
# plot results for budget saving model as a heatmap for the matrix of leaf and bounds privacy proportions
im = plt.imshow(budged_saving_accuracies, cmap='hot', interpolation='nearest')
plt.colorbar(im, label='test accuracy')
plt.xlabel('leaf privacy budget proportion')
plt.ylabel('bounds privacy budget proportion')
# make 4 x ticks and y ticks. I want to have the ticks on a equidistant grid on 1 to len(leaf_props)
ticks = np.array([10, 20, 30, 40])
ticks_labels = np.array([0.2, 0.4, 0.6, 0.8])
plt.xticks(ticks, ticks_labels)
plt.yticks(ticks, ticks_labels)

plt.savefig('Experiments/main_report_plots/budged_distribution_saving.pdf')
plt.show()





