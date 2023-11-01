import numpy as np
import federated_private_quantiles as fpq

class FederatedPrivateDecisionTree:
    def __init__(self, max_depth=5, 
                 min_samples_split=2, 
                 num_quantiles=10, 
                 privacy_budget=1, 
                 diff_privacy = True, 
                 leaf_mechanism='laplace', 
                 node_mechanism='laplace'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.num_quantiles = num_quantiles
        self.privacy_budget = privacy_budget
        self.diff_privacy = diff_privacy
        self.leaf_mechanism = leaf_mechanism
        self.node_mechanism = node_mechanism
        self.tree = {}
        self.quantiles = {}

    def __entropy(self, p):
        if p == 0 or p == 1:
            return 0
        return -p * np.log2(p)
    
    def __distribute_privacy_budget(self):
        self.quantile_epsilon = 0.5 * self.privacy_budget / ((self.max_depth + 1) * self.num_quantiles)
        self.node_epsilon = 0.5 * self.privacy_budget / (self.max_depth + 1)
        self.leaf_epsilon = 0.5 * self.privacy_budget
        return 

    def __quantile_calculator(self, X):
        n_samples, n_features = X.shape
        # Define all possible quantile probabilities
        quantile_probabilities = np.array([i / n_samples for i in range(1, n_samples)])
        # Select num_quantiles quantile probabilities from the possible quantile probabilities in an equidistant manner
        quantile_probabilities = quantile_probabilities[::int(n_samples / self.num_quantiles)]
        for feature in range(n_features):
            # Check if feature is binary, hence has only 2 values (categorical feature)
            if len(np.unique(X[:, feature])) == 2:
                # make quantile as middlepoint between the two values
                self.quantiles[feature] = np.array([np.mean(np.unique(X[:, feature]))])
                continue
            self.quantiles[feature] = fpq.FederatedPrivateQuantiles(quantile_probabilities, X[:, feature], n_samples)
            self.quantiles[feature] = self.quantiles[feature].compute_quantiles().z
        return 
    
    def __create_leaf(self, y):
        bincounts = np.bincount(y)
        if (self.diff_privacy == False):
            leaf_value = int(np.argmax(bincounts))
            return leaf_value
        if (self.diff_privacy == True):
            # Add laplace noise to the bincounts
            if (self.leaf_mechanism == 'laplace'):
                bincounts = np.array([np.random.laplace(x, 1 / self.leaf_epsilon) for x in bincounts])
                leaf_value = int(np.argmax(bincounts))
                return leaf_value
            if (self.leaf_mechanism == 'exponential'):
                # Use number of samples per class as score for each class for the exponential mechanism
                scores = bincounts
                # sensitivity is 1 as removing/adding one datapoint changes result by 1
                sensitivity = 1
                probabilities = np.exp(self.leaf_epsilon * scores / (2 * sensitivity))
                probabilities = probabilities / np.sum(probabilities)
                leaf_value = int(np.random.choice(np.arange(len(bincounts)), p=probabilities))
                return leaf_value

            
    

    def __quantile_binning(self, X, feature):
        # Bin the feature into a histogram according to the quantiles and store the bin counts in each bin
        feature_binned = np.digitize(X[:, feature], bins=self.quantiles[feature])
        feature_bincounts = np.bincount(feature_binned)
        if (self.diff_privacy == False or self.node_mechanism == 'exponential_mechanism'):
            return feature_bincounts
        if (self.node_mechanism == 'laplace'):
            # Add laplace noise to each bincount of the histogram
            feature_bincounts = np.array([np.random.laplace(x, 1 / self.node_epsilon) for x in feature_bincounts])
            return feature_bincounts
    
    def __best_split(self, X, y):
        n_samples, n_features = X.shape
        best_entropy = np.inf
        best_feature = None
        best_threshold = None
        # create matrix to store entropy for each split
        entropy_matrix = np.zeros((n_features, self.num_quantiles))
        threshold_counter = -1
        for feature in range(n_features):
            # Bin the numerical feature according to the quantiles. 
            feature_bincounts = self.__quantile_binning(X, feature)
            for threshold in self.quantiles[feature]:
                threshold_counter += 1
                # Calculate how many points are left of the threshold using the binned feature and round to closest integer
                left = round(np.sum(feature_bincounts[:np.digitize(threshold, bins=self.quantiles[feature])]))
                if left <= 0: # Go to next iteration, no split here (all points are on the right side of the threshold)
                    if self.node_mechanism == 'exponential':
                        entropy_matrix[feature, threshold_counter] = np.inf
                    continue
                right = round(np.sum(feature_bincounts[np.digitize(threshold, self.quantiles[feature]):]))
                if right <= 0: # Go to next iteration, no split here (all points are on the left side of the threshold)
                    if self.node_mechanism == 'exponential':
                        entropy_matrix[feature, threshold_counter] = np.inf
                    continue

                # Calculate the proportions of the labels in the left and right split
                p_left = sum(y[X[:, feature] < threshold]) / left
                p_right = sum(y[X[:, feature] >= threshold]) / right
                # Adjust proportions to be in [0, 1] (due to noise added to bincounts)
                if p_left > 1:
                    p_left = 1
                if p_right > 1:
                    p_right = 1

                # Calculate entropy of the splitted space (want to minimize entropy to maximize impurity decrease)
                entropy = (left/n_samples)*self.__entropy(p_left) + (right/n_samples)*self.__entropy(p_right)

                # If all points are on one side of the threshold, don't consider this split (maybe add a different solution later with min_leaf_size) and directly 
                # create a leaf when this happens (in the grow_tree function)
                if np.sum(X[:, feature] < threshold) == 0 or np.sum(X[:, feature] >= threshold) == 0:
                    if self.node_mechanism == 'exponential':
                        entropy_matrix[feature, threshold_counter] = np.inf
                    continue
                
                if self.node_mechanism == 'exponential':
                    # Add entropy to entropy matrix
                    entropy_matrix[feature, threshold_counter] = entropy

                if self.node_mechanism == 'laplace':
                    if entropy < best_entropy:
                        best_entropy = entropy
                        best_feature = feature
                        best_threshold = threshold  
        if self.node_mechanism == 'exponential':
            # Find best split according to exponential mechanism with score function as negative entropy
            scores = -entropy_matrix
            # Calculate sensitivity
            # TBD

        return best_entropy, best_feature, best_threshold


    def fit(self, X, y):
        if (self.diff_privacy == True):
            # Distribute privacy budget
            self.__distribute_privacy_budget()
        # Calculate quantiles
        self.__quantile_calculator(X)
        # Grow tree
        self.tree = self.__grow_tree(X, y, depth=0)

    def __grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        # Check termination conditions to potentially create a leaf
        if (depth >= self.max_depth or n_samples < self.min_samples_split or n_labels == 1):
            return self.__create_leaf(y)
        
        # Find best split
        best_entropy, best_feature, best_threshold = self.__best_split(X, y)

        # If didn't find a split that reduces entropy, make a leaf node
        if best_entropy == np.inf:
            return self.__create_leaf(y)
        
        # Otherwise divide space to create sub-trees
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = X[:, best_feature] >= best_threshold

        # Grow subtrees
        left_tree = self.__grow_tree(X[left_idxs], y[left_idxs], depth+1)
        right_tree = self.__grow_tree(X[right_idxs], y[right_idxs], depth+1)

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
        
