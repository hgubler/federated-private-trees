import sys
sys.path.append('') # Adds higher directory to python modules path
from experiments.evaluation_functions import accuracies_calc
from matplotlib import pyplot as plt
import numpy as np


#######################################################################################################
####################################### Low data volume setting #######################################
#######################################################################################################

####################################### High correlation setting #######################################
n_samples = 100
privacy_budgeds = np.linspace(0.1, 10, 10)
rho_1 = 0.9
rho_2 = 0.9
min_samples_leaf = 10
num_bins = 10
use_quantiles = False
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
max_depth_list = [5, 10]
fig, ax = plt.subplots(ncols=len(n_features_list), nrows=len(max_depth_list), sharex=True, sharey=True, figsize=(14, 7.25))

# Create lines for the legend
line1, = ax[0, 0].plot([], [], '-', label='differential privacy without budget saving', color='red')
line2, = ax[0, 0].plot([], [], '-', label='differential privacy with budget saving', color='orange')
line3, = ax[0, 0].plot([], [], '-', label='local decision tree', color='green')
line4, = ax[0, 0].plot([], [], '-', label='centralized decision tree', color='blue')

for max_depth in max_depth_list:
    for n_features in n_features_list:
        mean_1 = np.zeros(n_features)
        mean_2 = np.ones(n_features)
        accuracy, local_accuracy, normal_dt_accuracy, dp_saving_accuracy = accuracies_calc(privacy_budgeds=privacy_budgeds,
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
                                                                     party_idx=party_idx,
                                                                     mixture_model=mixture_model,
                                                                     n_centers=n_centers,
                                                                     seed=seed)
        ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].plot(privacy_budgeds, accuracy, '-o', color='red')
        ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].plot(privacy_budgeds, dp_saving_accuracy, '-o', color='orange')
        ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].plot(privacy_budgeds, local_accuracy, '-o', color='green')
        ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].plot(privacy_budgeds, normal_dt_accuracy, '-o', color='blue')
        ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].set_title('n_features={}, max_depth={}'.format(n_features, max_depth))
        if (n_features == n_features_list[0]):
            ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].set_ylabel('test accuracy')
        if (max_depth == max_depth_list[-1]):
            ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].set_xlabel('privacy budget')
        print((1 + n_features_list.index(n_features)) * (1 + max_depth_list.index(max_depth)))
fig.legend(handles=[line1, line2, line3, line4], loc='upper center', bbox_to_anchor=(0.515, 0.135))
fig.subplots_adjust(bottom=0.2) 
fig.suptitle('High correlation setting (rho=0.9)')
plt.savefig('Experiments/appendix_plots/accuracy_vs_privacy_budged_high_corr_low_data.pdf', bbox_inches='tight')
print('done with high correlation setting!')
#######################################################################################################



####################################### Low correlation setting #######################################
rho_1 = 0.3
rho_2 = 0.3

fig2, ax = plt.subplots(ncols=len(n_features_list), nrows=len(max_depth_list), sharex=True, sharey=True, figsize=(14, 7.25))

# Create lines for the legend
line1, = ax[0, 0].plot([], [], '-', label='differential privacy without budget saving', color='red')
line2, = ax[0, 0].plot([], [], '-', label='differential privacy with budget saving', color='orange')
line3, = ax[0, 0].plot([], [], '-', label='local decision tree', color='green')
line4, = ax[0, 0].plot([], [], '-', label='centralized decision tree', color='blue')

for max_depth in max_depth_list:
    for n_features in n_features_list:
        mean_1 = np.zeros(n_features)
        mean_2 = np.ones(n_features)
        accuracy, local_accuracy, normal_dt_accuracy, dp_saving_accuracy = accuracies_calc(privacy_budgeds=privacy_budgeds,
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
                                                                     party_idx=party_idx,
                                                                     mixture_model=mixture_model,
                                                                     n_centers=n_centers,
                                                                     seed=seed)
        ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].plot(privacy_budgeds, accuracy, '-o', color='red')
        ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].plot(privacy_budgeds, dp_saving_accuracy, '-o', color='orange')
        ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].plot(privacy_budgeds, local_accuracy, '-o', color='green')
        ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].plot(privacy_budgeds, normal_dt_accuracy, '-o', color='blue')
        ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].set_title('n_features={}, max_depth={}'.format(n_features, max_depth))
        if (n_features == n_features_list[0]):
            ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].set_ylabel('test accuracy')
        if (max_depth == max_depth_list[-1]):
            ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].set_xlabel('privacy budget')
        print((1 + n_features_list.index(n_features)) * (1 + max_depth_list.index(max_depth)))
fig2.legend(handles=[line1, line2, line3, line4], loc='upper center', bbox_to_anchor=(0.515, 0.135))
fig2.subplots_adjust(bottom=0.2) 
fig2.suptitle('Low correlation setting (rho=0.3)')
plt.savefig('Experiments/appendix_plots/accuracy_vs_privacy_budged_low_corr_low_data.pdf', bbox_inches='tight')
print('done with low correlation setting!')

#######################################################################################################
####################################### Low data volume setting #######################################
#######################################################################################################






#######################################################################################################
####################################### High data volume setting #######################################
#######################################################################################################

####################################### High correlation setting #######################################
n_samples = 1000
party_idx = []
n_parties = 5
sample_per_party = n_samples / n_parties
if n_samples % n_parties != 0:
    raise ValueError('n_samples must be divisible by n_parties')
for i in range(n_parties):
    party_idx.append(np.arange(i * sample_per_party, (i + 1) * sample_per_party, dtype=int))
rho_1 = 0.9
rho_2 = 0.9

fig3, ax = plt.subplots(ncols=len(n_features_list), nrows=len(max_depth_list), sharex=True, sharey=True, figsize=(14, 7.25))

# Create lines for the legend
line1, = ax[0, 0].plot([], [], '-', label='differential privacy without budget saving', color='red')
line2, = ax[0, 0].plot([], [], '-', label='differential privacy with budget saving', color='orange')
line3, = ax[0, 0].plot([], [], '-', label='local decision tree', color='green')
line4, = ax[0, 0].plot([], [], '-', label='centralized decision tree', color='blue')

for max_depth in max_depth_list:
    for n_features in n_features_list:
        mean_1 = np.zeros(n_features)
        mean_2 = np.ones(n_features)
        accuracy, local_accuracy, normal_dt_accuracy, dp_saving_accuracy = accuracies_calc(privacy_budgeds=privacy_budgeds,
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
                                                                     party_idx=party_idx,
                                                                     mixture_model=mixture_model,
                                                                     n_centers=n_centers,
                                                                     seed=seed)
        ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].plot(privacy_budgeds, accuracy, '-o', color='red')
        ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].plot(privacy_budgeds, dp_saving_accuracy, '-o', color='orange')
        ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].plot(privacy_budgeds, local_accuracy, '-o', color='green')
        ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].plot(privacy_budgeds, normal_dt_accuracy, '-o', color='blue')
        ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].set_title('n_features={}, max_depth={}'.format(n_features, max_depth))
        if (n_features == n_features_list[0]):
            ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].set_ylabel('test accuracy')
        if (max_depth == max_depth_list[-1]):
            ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].set_xlabel('privacyt budget')
        print((1 + n_features_list.index(n_features)) * (1 + max_depth_list.index(max_depth)))
fig3.legend(handles=[line1, line2, line3, line4], loc='upper center', bbox_to_anchor=(0.515, 0.135))
fig3.subplots_adjust(bottom=0.2) 
fig3.suptitle('High correlation setting (rho=0.9)')
plt.savefig('Experiments/appendix_plots/accuracy_vs_privacy_budged_high_corr_high_data.pdf', bbox_inches='tight')
print('done with high correlation setting!')

#######################################################################################################

####################################### Low correlation setting #######################################

rho_1 = 0.3
rho_2 = 0.3

fig4, ax = plt.subplots(ncols=len(n_features_list), nrows=len(max_depth_list), sharex=True, sharey=True, figsize=(14, 7.25))

# Create lines for the legend
line1, = ax[0, 0].plot([], [], '-', label='differential privacy without budget saving', color='red')
line2, = ax[0, 0].plot([], [], '-', label='differential privacy with budget saving', color='orange')
line3, = ax[0, 0].plot([], [], '-', label='local decision tree', color='green')
line4, = ax[0, 0].plot([], [], '-', label='centralized decision tree', color='blue')

for max_depth in max_depth_list:
    for n_features in n_features_list:
        mean_1 = np.zeros(n_features)
        mean_2 = np.ones(n_features)
        accuracy, local_accuracy, normal_dt_accuracy, dp_saving_accuracy = accuracies_calc(privacy_budgeds=privacy_budgeds,
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
                                                                     party_idx=party_idx,
                                                                     mixture_model=mixture_model,
                                                                     n_centers=n_centers,
                                                                     seed=seed)
        ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].plot(privacy_budgeds, accuracy, '-o', color='red')
        ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].plot(privacy_budgeds, dp_saving_accuracy, '-o', color='orange')
        ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].plot(privacy_budgeds, local_accuracy, '-o', color='green')
        ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].plot(privacy_budgeds, normal_dt_accuracy, '-o', color='blue')
        ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].set_title('n_features={}, max_depth={}'.format(n_features, max_depth))
        if (n_features == n_features_list[0]):
            ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].set_ylabel('test accuracy')
        if (max_depth == max_depth_list[-1]):
            ax[max_depth_list.index(max_depth), n_features_list.index(n_features)].set_xlabel('privacy budget')
        print((1 + n_features_list.index(n_features)) * (1 + max_depth_list.index(max_depth)))
fig4.legend(handles=[line1, line2, line3, line4], loc='upper center', bbox_to_anchor=(0.515, 0.135))
fig4.subplots_adjust(bottom=0.2) 
fig4.suptitle('Low correlation setting (rho=0.3)')
plt.savefig('Experiments/appendix_plots/accuracy_vs_privacy_budged_low_corr_high_data.pdf', bbox_inches='tight')
print('done with low correlation setting!')