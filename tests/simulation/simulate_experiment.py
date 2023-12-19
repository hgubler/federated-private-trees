import sys
sys.path.append('') # Adds higher directory to python modules path
from algorithms.federated_private_decision_tree_v2 import FederatedPrivateDecisionTree
from tests.simulation.simulate_data import SimulateData
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

class SimulateExperiment:
    def __init__(self, 
                 privacy_budgeds, 
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
        self.privacy_budgeds = privacy_budgeds
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
        self.seed = seed
        self.dp_accuracies = []
        self.no_dp_accuracies = []
        self.sklearn_accuracies = []

        # paralellize calculate_accuracies
        self.calculate_accuracies()
        