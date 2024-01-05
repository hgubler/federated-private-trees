import sys
sys.path.append('') # Adds higher directory to python modules path
from FedDifferentialPrivacyTree.federated_private_decision_tree import FederatedPrivateDecisionTree
from FedDifferentialPrivacyTree.decision_tree import DecisionTree
from experiments.data.simulate_data import SimulateData
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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
                    party_idx = [],
                    mixture_model = False,
                    n_centers = 0,
                    seed=0):
    """
    This function calculates the accuracies of the federated private decision trees,
    the local decision tree and centralized decision tree for different privacy budgets on synthetic data.

    Parameters
    ----------
    privacy_budgeds : array-like
        The privacy budgets to be tested.
    max_depth : int, optional
        The maximum depth of the decision tree. The default is 4.
    num_bins : int, optional
        The number of bins to be used for binning the features in our methods. The default is 10.
    min_samples_leaf : int, optional
        The minimum number of samples in a leaf. The default is 10.
    use_quantiles : bool, optional
        Whether to use quantiles for the histogram. The default is False. (Experimental)
    n_rep : int, optional
        The number of repetitions. The default is 100.
    rho_1 : float, optional
        The correlation scenario of the features for the first class. The default is 0.5.
    rho_2 : float, optional
        The correlation scenario of the features for the second class. The default is 0.9.
    n_samples : int, optional
        The number of samples. The default is 1000.
    n_features : int, optional
        The number of features. The default is 10.
    party_idx : list, optional
        The indices of the samples of each party. The default is [].
    mixture_model : bool, optional
        Whether to use a mixture model for the synthetic data. The default is False.
    n_centers : int, optional
        The number of centers for the mixture model. The default is 0.
    seed : int, optional
        The seed for the experiment. The default is 0.

    Returns
    -------
    dp_accuracies : array-like
        The accuracies of the federated private decision tree without budget saving.
    local_accuracies : array-like
        The accuracies of the local decision tree.
    normal_dt_accuracies : array-like
        The accuracies of the centralized decision tree.
    dp_saving_accuracies : array-like
        The accuracies of the federated private decision tree with budget saving.
    """

    # initialize arrays
    dp_accuracies = []
    local_accuracies = []
    normal_dt_accuracies = []
    dp_saving_accuracies = []
    for i in range(privacy_budgeds.shape[0]):
        accuracy = 0
        local_accuracy = 0
        normal_dt_accuracy = 0
        dp_saving_accuracy = 0
        for j in range(n_rep):
            # initialize models
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
                                             save_budget=False)
            dp_saving_model = FederatedPrivateDecisionTree(max_depth=max_depth, 
                                             min_samples_leaf=min_samples_leaf, 
                                             privacy_budget=privacy_budgeds[i], 
                                             diff_privacy=True, num_bins=num_bins, 
                                             use_quantiles=use_quantiles,
                                             party_idx=party_idx,
                                             save_budget=True,
                                             )
            # generate data
            simulate_data = SimulateData(n_features, mean_1, mean_2, rho_1, rho_2, 2 * n_samples, mixture_model=mixture_model, n_centers=n_centers, seed=seed)
            X, y = simulate_data.generate_data()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
            # split data for local model
            X_party_1 = X_train[party_idx[0]]
            y_party_1 = y_train[party_idx[0]]
            # fit models
            model.fit(X_train, y_train)
            local_model.fit(X_party_1, y_party_1)
            normal_dt_model.fit(X_train, y_train)
            dp_saving_model.fit(X_train, y_train)
            # make predictions
            predictions = model.predict(X_test)
            local_predictions = local_model.predict(X_test)
            normal_dt_predictions = normal_dt_model.predict(X_test)
            dp_saving_predictions = dp_saving_model.predict(X_test)
            # calculate accuracies
            accuracy = accuracy + accuracy_score(y_test, predictions)
            local_accuracy = local_accuracy + accuracy_score(y_test, local_predictions)
            normal_dt_accuracy = normal_dt_accuracy + accuracy_score(y_test, normal_dt_predictions)
            dp_saving_accuracy = dp_saving_accuracy + accuracy_score(y_test, dp_saving_predictions)
            seed = seed + 1
        # calculate mean accuracies
        accuracy = accuracy / n_rep
        local_accuracy = local_accuracy / n_rep
        dp_saving_accuracy = dp_saving_accuracy / n_rep
        normal_dt_accuracy = normal_dt_accuracy / n_rep
        # append accuracies
        dp_accuracies.append(accuracy)
        local_accuracies.append(local_accuracy)
        normal_dt_accuracies.append(normal_dt_accuracy)
        dp_saving_accuracies.append(dp_saving_accuracy)
    return dp_accuracies, local_accuracies, normal_dt_accuracies, dp_saving_accuracies


def no_budged_saving_accuracy(leaf_prop, n_rep, n_samples, n_features, n_centers, mixture_model, rho_1, rho_2, privacy_budget, seed):
    """
    This function computes the accuracy of a non-budget saving model with the possibility to vary 
    the proportion of the privacy budget that is used for the leaf nodes.

    Parameters
    ----------
    leaf_prop : float
        The proportion of the privacy budget that is used for the leaf nodes. 
    n_rep : int
        The number of repetitions of the experiment.
    n_samples : int
        The number of samples.
    n_features : int
        The number of features.
    n_centers : int
        The number of centers for the mixture model.
    mixture_model : bool
        Whether to use a mixture model or not.
    rho_1 : float
        The correlatio scenario between the features of the first class.
    rho_2 : float
        The correlation scenario between the features of the second class.
    privacy_budget : int
        The privacy budget.
    seed : int
        The random seed.

    Returns
    -------
    no_budget_saving_accuracy : float
        The accuracy of the non-budget saving model.
    """
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
    """
    This function computes the accuracy of a budget saving model with the possibility to vary
    the proportion of the privacy budget that is used for the leaf nodes and the bounds, as well as
    the proportion of the privacy budget that is used for the impurity bounds.

    Parameters
    ----------
    leaf_prop : float
        The proportion of the privacy budget that is used for the leaf nodes.
    bounds_prop : float
        The proportion of the privacy budget that is used for the impurity bounds.
    n_rep : int
        The number of repetitions of the experiment.
    n_samples : int
        The number of samples.
    n_features : int
        The number of features.
    n_centers : int
        The number of centers for the mixture model.
    mixture_model : bool
        Whether to use a mixture model or not.
    rho_1 : float
        The correlatio scenario between the features of the first class.
    rho_2 : float
        The correlation scenario between the features of the second class.
    privacy_budget : int
        The privacy budget.
    seed : int
        The random seed.

    Returns
    -------
    budget_saving_accuracy : float
        The accuracy of the budget saving model.
    """
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