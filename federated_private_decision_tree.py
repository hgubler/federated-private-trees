import numpy as np
import federated_private_quantiles as fpq

class FederatedPrivateDecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = {}

    def entropy(self, y):
        p = np.sum(y) / len(y)
        return -p*np.log2(p) - (1-p)*np.log2(1-p)

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check termination conditions
        if (depth >= self.max_depth or n_samples < self.min_samples_split or n_labels == 1):
            leaf_value = np.argmax(np.bincount(y))
            return leaf_value

        # Find best split
        best_entropy = np.inf
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            quantiles = fpq.FederatedPrivateQuantiles(np.linspace(0, 1, num=11), X[:, feature], n_samples)
            quantiles, _ = quantiles.compute_quantiles().z
            for threshold in quantiles:
                y_left = y[X[:, feature] < threshold]
                y_right = y[X[:, feature] >= threshold]
                entropy = (len(y_left)/n_samples)*self.entropy(y_left) + (len(y_right)/n_samples)*self.entropy(y_right)
                if entropy < best_entropy:
                    best_entropy = entropy
                    best_feature = feature
                    best_threshold = threshold

        # Split data
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = X[:, best_feature] >= best_threshold

        # Grow subtrees
        left_tree = self._grow_tree(X[left_idxs], y[left_idxs], depth+1)
        right_tree = self._grow_tree(X[right_idxs], y[right_idxs], depth+1)

        # Create node dictionary
        node = {'feature': best_feature,
                'threshold': best_threshold,
                'left': left_tree,
                'right': right_tree}

        return node

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if isinstance(node, int):
            return node
        feature_val = x[node['feature']]
        if feature_val < node['threshold']:
            return self._traverse_tree(x, node['left'])
        else:
            return self._traverse_tree(x, node['right'])