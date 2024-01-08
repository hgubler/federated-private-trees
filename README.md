# Federated Differentially Private Decision Tree

This repository contains the code for a semester project focused on designing a federated differentially private decision tree algorithm.

## Project Structure

The project is structured into two main directories: `FedDifferentialPrivacyTree` and `Experiments`.

### Algorithms

The `FedDifferentialPrivacyTree` directory contains the implementation of the federated differentially private decision tree algorithm. The main file in this directory is `federated_private_decision_tree.py`.

This algorithm can be run with and without privacy budget saving, controlled by the `save_budget` parameter. When `save_budget` is set to `True`, the algorithm uses feature impurity bounds to save privacy budget.

### Experiments

The `Experiments` directory contains the code used to create the plots for the simulations, as detailed in the project report. This includes various scripts for running the algorithm under different conditions and visualizing the results.


## Installation

To install the required dependencies, run the following command:

```bash
pip install 'git+https://github.com/hgubler/federated-private-trees'
```

## Usage

The algorithm can be run as follows, for suitable values of the parameters and data:

```python
from FedDifferentialPrivacyTree.federated_private_decision_tree import FederatedPrivateDecisionTree as FPDT
tree = FDPT(max_depth, 
            min_samples_leaf, 
            num_bins, 
            privacy_budget, 
            leaf_privacy_proportion, 
            bounds_privacy_proportion, 
            save_budget,
            party_idx)
tree.fit(X_train, y_train)   
tree.predict(X_test)     
```
