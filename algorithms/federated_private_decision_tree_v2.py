import numpy as np
from algorithms.private_binner import PrivateBinner

class FederatedPrivateDecisionTree:
    def __init__(self, max_depth=5, 
                 min_samples_split=2, 
                 num_bins=10, 
                 privacy_budget=1, 
                 diff_privacy = True):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.num_bins = num_bins
        self.privacy_budget = privacy_budget
        self.diff_privacy = diff_privacy
        self.tree = {}
        self.quantiles = {}


    def __entropy(self, p):
        if p == 0 or p == 1:
            return 0
        return -p * np.log2(p)
    
    def __create_leaf(self, X0_binned, X1_binned):
        # take the column sums of X0_binned and X1_binned
        label_0 = np.sum(X0_binned, axis=0).mean()
        label_1 = np.sum(X1_binned, axis=0).mean()
        # return the label with the highest count
        if (label_0 > label_1):
            return 0
        else:
            return 1
    
    
    def __best_split(self, X0_binned, X1_binned, num_bins):
        n_features = X0_binned.shape[1]
        best_entropy = np.inf
        best_feature = None
        best_threshold = None
        for feature in range(n_features):
            for threshold in range(num_bins):
                # Calculate how many points are left of the threshold using the binned feature and round to closest integer
                left = np.sum(X0_binned[:threshold, feature]) + np.sum(X1_binned[:threshold, feature])
                if left <= 0: # Go to next iteration, no split here (all points are on the right side of the threshold)
                    continue
                right = np.sum(X0_binned[threshold:, feature]) + np.sum(X1_binned[threshold:, feature])
                if right <= 0: # Go to next iteration, no split here (all points are on the left side of the threshold)
                    continue

                # Calculate the proportions of the labels in the left and right split
                p_left = np.sum(X1_binned[:threshold, feature]) / left
                p_right = np.sum(X1_binned[threshold:, feature]) / right
                # Adjust proportions to be in [0, 1] (due to noise added to bincounts)
                if p_left > 1:
                    p_left = 1
                if p_left < 0:
                    p_left = 0
                if p_right > 1:
                    p_right = 1
                if p_right < 0:
                    p_right = 0

                # Calculate entropy of the splitted space (want to minimize entropy to maximize impurity decrease)
                entropy = left*self.__entropy(p_left) + right*self.__entropy(p_right)

                if entropy < best_entropy:
                    best_entropy = entropy
                    best_feature = feature
                    best_threshold = threshold

        return best_entropy, best_feature, best_threshold


    def fit(self, X, y):
        # Grow tree
        self.tree = self.__grow_tree(X, y, num_bins=self.num_bins, depth=0)

    def __grow_tree(self, X, y, num_bins, depth=0):
        # bin the features separated by label
        X_0 = X[y == 0]
        X_1 = X[y == 1]
        # calculate binned X_0 and X_1
        binner = PrivateBinner(num_bins = num_bins, privacy_budget=self.privacy_budget / self.max_depth)
        X0_binned = binner.fit_transform(X_0)
        X1_binned = binner.fit_transform(X_1)
        # Check termination conditions to potentially create a leaf
        if (depth >= self.max_depth):
            return self.__create_leaf(X0_binned, X1_binned)
        
        # Find best split
        best_entropy, best_feature, best_threshold = self.__best_split(X0_binned, X1_binned, num_bins)

        # If didn't find a split that reduces entropy, make a leaf node
        if best_entropy == np.inf:
            return self.__create_leaf(X0_binned, X1_binned)
        
        # Otherwise divide space to create sub-trees
        # Find the value in the original data that corresponds to the threshold
        # (we need to do this because we binned the data)
        best_threshold = np.quantile(X[:, best_feature], best_threshold / num_bins)
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = X[:, best_feature] >= best_threshold

        if int(left_idxs.sum() * self.num_bins) == 0 or int(right_idxs.sum() * self.num_bins) == 0:
            return self.__create_leaf(X0_binned, X1_binned)
    
        # Grow subtrees
        left_tree = self.__grow_tree(X=X[left_idxs], 
                                     y=y[left_idxs], 
                                     num_bins=int(left_idxs.sum() * self.num_bins), 
                                     depth=depth+1)
        right_tree = self.__grow_tree(X=X[right_idxs], 
                                      y=y[right_idxs], 
                                      num_bins=int(right_idxs.sum() * self.num_bins), 
                                      depth=depth+1)

        # Create node 
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
        
