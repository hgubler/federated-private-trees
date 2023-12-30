import numpy as np

class DecisionTree:
    def __init__(self, 
                 max_depth=5, 
                 min_samples_leaf=10):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        


    def __entropy(self, p):
        return p * (1-p)
    
    def __create_leaf(self, y):
        return int(np.bincount(y).argmax())
    
    def __best_split(self, X, y):
        n_features = X.shape[1]
        best_entropy = np.inf
        best_feature = None
        best_threshold_index = None
        for feature in range(n_features):
            for threshold in X[:, feature]:
                # Calculate how many points are left of the threshold 
                left = np.sum(X[:, feature] < threshold)
                if left == 0: # Go to next iteration, no split here (all points are on the right side of the threshold)
                    continue
                right = np.sum(X[:, feature] >= threshold)
                if right == 0: # Go to next iteration, no split here (all points are on the left side of the threshold)
                    continue

                # Calculate the proportions of the labels in the left and right split
                p_left = np.sum(y[X[:, feature] < threshold] == 0) / left
                p_right = np.sum(y[X[:, feature] >= threshold] == 0) / right

                # Calculate entropy of the splitted space (want to minimize entropy to maximize impurity decrease)
                entropy = left*self.__entropy(p_left) + right*self.__entropy(p_right)

                if entropy < best_entropy:
                    best_entropy = entropy
                    best_feature = feature
                    best_threshold = threshold

        return best_entropy, best_feature, best_threshold

    def fit(self, X, y):
        # Grow tree
        self.tree = self.__grow_tree(X, y, depth=0)

    def __grow_tree(self, X, y, depth=0):
        # Check termination conditions to potentially create a leaf
        if (depth >= self.max_depth):
            return self.__create_leaf(y)
        
        if y.size <= self.min_samples_leaf:
            return self.__create_leaf(y)
        
        if np.all(y == y[0]):
            return self.__create_leaf(y)
        
        best_entropy, best_feature, best_threshold = self.__best_split(X, y)

        # If didn't find a split that reduces entropy, make a leaf node
        if best_entropy == np.inf:
            return self.__create_leaf(y)
        
        # Otherwise divide space to create sub-trees
        
        # Split data
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = X[:, best_feature] >= best_threshold


    
        # Grow subtrees
        left_tree = self.__grow_tree(X=X[left_idxs], 
                                     y=y[left_idxs],
                                     depth=depth+1)
        right_tree = self.__grow_tree(X=X[right_idxs], 
                                      y=y[right_idxs],
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
        