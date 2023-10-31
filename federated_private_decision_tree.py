import numpy as np
import federated_private_quantiles as fpq

class FederatedPrivateDecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, num_quantiles=10, privacy_budget=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.num_quantiles = num_quantiles
        self.privacy_budget = privacy_budget
        self.tree = {}
        self.quantiles = {}

    def entropy(self, p):
        return -p * np.log2(p)
    
    def distribute_privacy_budget(self, epsilon):
        self.quantile_epsilon = 0.5 * epsilon / ((self.max_depth + 1) * self.num_quantiles)
        self.node_epsilon = 0.5 * epsilon / (self.max_depth + 1)
        self.leaf_epsilon = 0.5 * epsilon

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # distribute privacy budget
        self.distribute_privacy_budget(self.privacy_budget)
        # define all possible quantile probabilities
        quantile_probabilities = np.array([i / n_samples for i in range(1, n_samples)])
        # select num_quantiles quantile probabilities from the possible quantile probabilities in an equidistant manner
        quantile_probabilities = quantile_probabilities[::int(n_samples / self.num_quantiles)]
        self.quantiles = {}
        for feature in range(n_features):
            self.quantiles[feature] = fpq.FederatedPrivateQuantiles(quantile_probabilities, X[:, feature], n_samples)
            self.quantiles[feature] = self.quantiles[feature].compute_quantiles().z
        self.tree = self._grow_tree(X, y, quantiles=self.quantiles)

    def _grow_tree(self, X, y, depth=0, quantiles=None):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check termination conditions
        if (depth >= self.max_depth or n_samples < self.min_samples_split or n_labels == 1):
            bincounts = np.bincount(y)
            # add laplace noise to the bincounts
            bincounts = np.array([np.random.laplace(x, 1 / self.leaf_epsilon) for x in bincounts])
            leaf_value = int(np.argmax(bincounts))
            return leaf_value

        best_entropy = np.inf
        best_feature = None
        best_threshold = None

        if quantiles is None:
            quantiles = self.quantiles
        for feature in range(n_features):
            # bin the feature into a histogram according to the quantiles and store the bin counts in each bin
            feature_binned = np.digitize(X[:, feature], bins=quantiles[feature])
            feature_bincounts = np.bincount(feature_binned)
            # add laplace noise to each bincount of the histogram
            feature_bincounts = np.array([np.random.laplace(x, 1 / self.node_epsilon) for x in feature_bincounts])
            for threshold in quantiles[feature]:
                # calculate y_left by summing up the bincounts of the bins that are left and round to closest integer
                left = round(np.sum(feature_bincounts[:np.digitize(threshold, bins=quantiles[feature])]))
                # change left to 1 if it is smaller than 1
                if left <= 0:
                    # go to next iteration
                    continue
                right = round(np.sum(feature_bincounts[np.digitize(threshold, quantiles[feature]):]))
                # change right to 1 if it is smaller than 1, no split here
                if right <= 0:
                    # go to next iteration
                    continue
                #y_left = y[X[:, feature] < threshold]
                #y_right = y[X[:, feature] >= threshold]
                # extract the smallest left elements from X[:, feature] and count how many of them are smaller than threshold
                p_left = sum(y[X[:, feature] < threshold]) / left
                p_right = sum(y[X[:, feature] >= threshold]) / right
                if p_left > 1:
                    p_left = 1
                if p_right > 1:
                    p_right = 1
                entropy = (left/n_samples)*self.entropy(p_left) + (right/n_samples)*self.entropy(p_right)
                if entropy < best_entropy:
                    best_entropy = entropy
                    best_feature = feature
                    best_threshold = threshold
        # if didnt find a split that reduces entropy, make a leaf node
        if best_entropy == np.inf:
            # make a leaf node
            bincounts = np.bincount(y)
            # add laplace noise to the bincounts
            bincounts = np.array([np.random.laplace(x, 1 / self.leaf_epsilon) for x in bincounts])
            leaf_value = int(np.argmax(bincounts))
            return leaf_value
        # Otherwise create sub-trees
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = X[:, best_feature] >= best_threshold

        # Grow subtrees
        left_tree = self._grow_tree(X[left_idxs], y[left_idxs], depth+1, quantiles)
        right_tree = self._grow_tree(X[right_idxs], y[right_idxs], depth+1, quantiles)

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
        
