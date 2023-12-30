import sys
sys.path.append('') # Adds higher directory to python modules path
from algorithms.federated_private_decision_tree_v2 import FederatedPrivateDecisionTree
from tests.simulation.simulate_data import SimulateData
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import multiprocessing
from functools import partial


class SimulateExperiment:
    def __init__(self, 
                 privacy_budgets, 
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
                 mixture_model = False,
                 n_centers = 0,
                 party_idx = [],
                 seed=0):
        self.privacy_budgets = privacy_budgets
        self.mean_1 = mean_1
        self.mean_2 = mean_2
        self.max_depth = max_depth
        self.num_bins = num_bins
        self.min_samples_leaf = min_samples_leaf
        self.use_quantiles = use_quantiles
        self.n_rep = n_rep
        self.rho_1 = rho_1
        self.rho_2 = rho_2
        self.n_samples = n_samples
        self.n_features = n_features
        self.save_budget = save_budget
        self.party_idx = party_idx
        self.mixutre_model = mixture_model
        self.n_centers = n_centers
        self.seed = seed


    def accuracies_calc(self, privacy_budget, random_state=0):
        random_state = random_state + self.seed
        dp_model = FederatedPrivateDecisionTree(max_depth=self.max_depth, 
                                                min_samples_leaf=self.min_samples_leaf, 
                                                privacy_budget=privacy_budget, 
                                                diff_privacy=True, 
                                                num_bins=self.num_bins, 
                                                use_quantiles=self.use_quantiles,
                                                save_budget=self.save_budget,
                                                party_idx=self.party_idx,
                                                random_state=random_state)
        
        no_dp_model = FederatedPrivateDecisionTree(max_depth=self.max_depth, 
                                                    min_samples_leaf=self.min_samples_leaf, 
                                                    diff_privacy=False, 
                                                    num_bins=self.num_bins, 
                                                    use_quantiles=self.use_quantiles)
        
        sklearn_model = DecisionTreeClassifier(max_depth=self.max_depth)

        
        simulate_data = SimulateData(self.n_features, self.mean_1, self.mean_2, self.rho_1, self.rho_2, 2 * self.n_samples, mixture_model=self.mixutre_model, n_centers=self.n_centers,  seed = random_state)
        X, y = simulate_data.generate_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=random_state)
        # fit models
        dp_model.fit(X_train, y_train)
        no_dp_model.fit(X_train, y_train)
        sklearn_model.fit(X_train, y_train)
        # make predictions
        dp_predictions = dp_model.predict(X_test)
        no_dp_predictions = no_dp_model.predict(X_test)
        sklearn_predictions = sklearn_model.predict(X_test)
        # calculate accuracies
        dp_accuracy = accuracy_score(y_test, dp_predictions)
        no_dp_accuracy = accuracy_score(y_test, no_dp_predictions)
        sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)
        return dp_accuracy, no_dp_accuracy, sklearn_accuracy

    def simulation_parallel(self, privacy_budget):
        pool = multiprocessing.Pool()
        accuracies_calc_with_args = partial(self.accuracies_calc, privacy_budget)
        accuracies = pool.map(accuracies_calc_with_args, range(self.n_rep))
        pool.close()
        pool.join()
        self.seed = self.seed + self.n_rep
        dp_accuracy = [acc[0] for acc in accuracies]
        no_dp_accuracy = [acc[1] for acc in accuracies]
        sklearn_accuracy = [acc[2] for acc in accuracies]
        dp_accuracy = np.sum(dp_accuracy) / self.n_rep
        no_dp_accuracy = np.sum(no_dp_accuracy) / self.n_rep
        sklearn_accuracy = np.sum(sklearn_accuracy) / self.n_rep
        return dp_accuracy, no_dp_accuracy, sklearn_accuracy

    def simulate_experiments(self):
        dp_accuracies = []
        no_dp_accuracies = []
        sklearn_accuracies = []
        for privacy_budget in self.privacy_budgets:
            dp_accuracy, no_dp_accuracy, sklearn_accuracy = self.simulation_parallel(privacy_budget)
            dp_accuracies.append(dp_accuracy)
            no_dp_accuracies.append(no_dp_accuracy)
            sklearn_accuracies.append(sklearn_accuracy)      
        return dp_accuracies, no_dp_accuracies, sklearn_accuracies
    



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n_samples = 100
    privacy_budgeds = np.linspace(0.1, 10, 10)
    rho_1 = 0.9
    rho_2 = 0.9
    min_samples_leaf = 10
    num_bins = 10
    use_quantiles = False
    save_budget = True
    seed = 0
    party_idx = []
    mixture_model = True
    n_centers = 5
    n_parties = 5
    sample_per_party = n_samples / n_parties
    if n_samples % n_parties != 0:
        raise ValueError('n_samples must be divisible by n_parties')
    for i in range(n_parties):
        party_idx.append(np.arange(i * sample_per_party, (i + 1) * sample_per_party, dtype=int))
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
            simulation = SimulateExperiment(privacy_budgets=privacy_budgeds,
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
            accuracy, no_dp_accuracy, sklearn_accuracy = simulation.simulate_experiments()
            ax[n_features_list.index(n_features), max_depth_list.index(max_depth)].plot(privacy_budgeds, accuracy, '-', color='red')
            ax[n_features_list.index(n_features), max_depth_list.index(max_depth)].plot(privacy_budgeds, no_dp_accuracy, '-', color='blue')
            ax[n_features_list.index(n_features), max_depth_list.index(max_depth)].plot(privacy_budgeds, sklearn_accuracy, '-', color='green')
            ax[n_features_list.index(n_features), max_depth_list.index(max_depth)].set_xlabel('privacy budged')
            ax[n_features_list.index(n_features), max_depth_list.index(max_depth)].set_title('n_features = {}, max_depth = {}'.format(n_features, max_depth))
            print((1 + n_features_list.index(n_features)) * (1 + max_depth_list.index(max_depth)))
    # set a general title for the figure specifying number of bins and number of samples
    fig.suptitle('Accuracy vs. privacy budged (number of bins = {}, number of samples = {})'.format(num_bins, n_samples))
    fig.legend(handles=[line1, line2, line3], loc='upper right')
    plt.show()





           