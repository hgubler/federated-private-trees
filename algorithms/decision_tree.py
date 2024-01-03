import numpy as np

class DecisionTree:
    """
    A class used to represent a Decision Tree.

    Attributes
    ----------
    max_depth : int
        The maximum depth of the tree. The default is 5.
    min_samples_leaf : int
        The minimum number of samples required to be at a leaf node. The default is 10.

    Methods
    -------
    fit(X, y):
        Trains the decision tree on the given data.
    predict(X):
        Makes predictions for the given data.
    __entropy(p):
        Calculates the entropy of a binary distribution.
    __create_leaf(y):
        Creates a leaf node based on the majority class in y.
    __best_split(X, y):
        Finds the best feature and threshold to split on.
    """

    def __init__(self, 
                 max_depth=5, 
                 min_samples_leaf=10):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        


    def __entropy(self, p):
        """
        Calculates the entropy of a binary distribution.

        Parameters
        ----------
            p : float
                The proportion of samples in one class.

        Returns
        -------
            float
                The entropy of the distribution.
        """
        return p * (1-p)
    
    def __create_leaf(self, y):
        """
        Creates a leaf node based on the majority class in y.

        Parameters
        ----------
            y : numpy array
                The labels of the samples at the node.

        Returns
        -------
            int
                The majority class in y.
        """
        return int(np.bincount(y).argmax())
    
    def __best_split(self, X, y):
        """
        Finds the best feature and threshold to split on.

        Parameters
        ----------
            X : numpy array
                The features of the samples at the node.
            y : numpy array
                The labels of the samples at the node.

        Returns
        -------
            tuple
                The best feature and threshold to split on.
        """
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
        """
        Trains the decision tree on the given data.

        Parameters
        ----------
            X : numpy array
                The features of the training data.
            y : numpy array
                The labels of the training data.
        """
        # Grow tree
        self.tree = self.__grow_tree(X, y, depth=0)

    def __grow_tree(self, X, y, depth=0):
        """
        Grows the decision tree recursively.

        Parameters
        ----------
            X : numpy array
                The features of the data at the current node.
            y : numpy array
                The labels of the data at the current node.
            depth : int
                The current depth of the tree.

        Returns
        -------
            dict
                The current node in the tree.
        """
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
        """
        Makes predictions for the given data.

        Parameters
        ----------
            X : numpy array
                The features of the data to make predictions for.

        Returns
        -------
            numpy array
                The predicted labels for the data.
        """
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        """
        Traverses the tree to make a prediction for a single sample.

        Parameters
        ----------
            x : numpy array
                The features of the sample to make a prediction for.
            node : dict
                The current node in the tree.

        Returns
        -------
            int
                The predicted label for the sample.
        """
        if isinstance(node, int):
            return node
        feature_val = x[node['feature']]
        if feature_val < node['threshold']:
            return self._traverse_tree(x, node['left'])
        else:
            return self._traverse_tree(x, node['right'])
        