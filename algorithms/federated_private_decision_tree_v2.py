import numpy as np
from algorithms.private_binner import PrivateBinner

class FederatedPrivateDecisionTree:
    def __init__(self, 
                 max_depth=5, 
                 min_samples_leaf=10, 
                 num_bins=10, 
                 privacy_budget=1, 
                 leaf_privacy_proportion=0.5,
                 bounds_privacy_proportion=0.25,
                 use_quantiles = False,
                 diff_privacy = True,
                 save_budget = False,
                 random_state = None,
                 party_idx = []):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.num_bins = num_bins
        self.privacy_budget = privacy_budget
        self.diff_privacy = diff_privacy
        self.use_quantiles = use_quantiles
        self.leaf_privacy_proportion = leaf_privacy_proportion
        self.bounds_privacy_proportion = bounds_privacy_proportion
        self.tree = {}
        self.save_budget = save_budget
        self.random_state = random_state
        self.party_idx = party_idx
        


    def __entropy(self, p):
        return p * (1-p)
    
    def __create_leaf(self, X, y, saved_budged=0):
        if self.diff_privacy == False:
            return int(np.bincount(y).argmax())
        if self.diff_privacy == True:
            leaf_epsilon = self.leaf_epsilon + saved_budged
            bincounts = np.bincount(y) + np.random.laplace(0, 1 / leaf_epsilon, size=2) # sensitivity 1
            if np.abs(bincounts[0] - bincounts[1]) < 0.3:
                return 0
            return int(np.argmax(bincounts))
    
    def __best_split(self, X0_binned, X1_binned, num_bins):
        n_features = X0_binned.shape[1]
        best_entropy = np.inf
        best_feature = None
        best_threshold_index = None
        for feature in range(n_features):
            for threshold in range(num_bins+1):
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
                n = left + right
                entropy = (left*self.__entropy(p_left) + right*self.__entropy(p_right)) / n

                if entropy < best_entropy:
                    best_entropy = entropy
                    best_entropy_return = n * entropy
                    best_feature = feature
                    best_threshold_index = threshold
        if best_entropy == np.inf:
            best_entropy_return = np.inf

        return best_entropy_return, best_feature, best_threshold_index
    
    def __best_split_save_budged(self, X, y, saved_budged, party_idx):
        n_features = X.shape[1]
        n_parties = len(self.party_idx)
        feature_entropy_bounds = np.zeros(n_features)
        for feature in range(n_features):
            for party in range(n_parties):
                entropy = self.__best_impurity_locally(X, y, feature, party, party_idx)
                feature_entropy_bounds[feature] = feature_entropy_bounds[feature] + entropy
        budget = self.node_epsilon + saved_budged
        epsilon_bounds = budget * self.bounds_privacy_proportion
        epsilon_node = budget * (1 - self.bounds_privacy_proportion)
        saved_budged = 0
        sensitivity = 5 / 4
        feature_entropy_bounds = feature_entropy_bounds + np.random.laplace(0, sensitivity / epsilon_bounds, size=n_features)
        # sort them
        sorted_idx = np.argsort(feature_entropy_bounds)
        
        # Create an array of the lenght of the number of features where every entry is True
        # This array will be used to keep track of which features are still possible to split on
        min_samples_leaf = True
        min_samples_class = True
        best_entropy = np.inf
        best_feature = None
        best_threshold_index = None
        possible_features = np.ones(n_features, dtype=bool)
        for feature in sorted_idx:
            if possible_features[feature] == False:
                saved_budged = saved_budged + epsilon_node / n_features
                continue
            # bin the feature separated by label
            X_0 = X[y == 0, feature]
            X_1 = X[y == 1, feature]

            bins = np.linspace(np.min(X[:, feature]), np.max(X[:, feature]), self.num_bins+1, endpoint=True)
            bins = bins.reshape(-1, 1)
            binner = PrivateBinner(bins=bins, privacy_budget=epsilon_node / n_features, diff_privacy=self.diff_privacy)
            X0_binned = binner.fit_transform(X_0)
            X1_binned = binner.fit_transform(X_1)
            
            if min_samples_leaf:
                n_samples = np.sum(X0_binned, axis=0) + np.sum(X1_binned, axis=0)
                if n_samples > self.min_samples_leaf:
                    min_samples_leaf = False

            if min_samples_class:
                n_samples_0 = np.sum(X0_binned, axis=0)
                n_samples_1 = np.sum(X1_binned, axis=0)
                if n_samples_0 > 0 and n_samples_1 > 0:
                    min_samples_class = False

            # Find best split
            entropy, _, threshold_index = self.__best_split(X0_binned, X1_binned, self.num_bins)
            if entropy < best_entropy:
                best_entropy = entropy
                best_feature = feature
                best_threshold_index = threshold_index
                # delete variables that have entropy bounds higher than the best entropy
                for feature in sorted_idx:
                    if feature_entropy_bounds[feature] > best_entropy:
                        possible_features[feature] = False


        # print("saved budged: ", saved_budged)
        if min_samples_leaf or min_samples_class:
            best_entropy = np.inf # creates a leaf if this is the case
        return best_entropy, best_feature, best_threshold_index, saved_budged

    def __best_impurity_locally(self, X, y, feature, party, party_idx):
        X_party = X[party_idx[party], feature]
        y_party = y[party_idx[party]]
        best_entropy = np.inf
        for threshold in X_party:
            left = np.sum(y_party[X_party < threshold])
            right = np.sum(y_party[X_party >= threshold])
            if left + right == 0:
                continue
            p_left = left / (left + right)
            p_right = right / (left + right)
            entropy = left*self.__entropy(p_left) + right*self.__entropy(p_right)
            if entropy < best_entropy:
                best_entropy = entropy   
        return best_entropy


    def fit(self, X, y):
        self.__distribute_privacy_budged()
        # Grow tree$
        if self.random_state != None:
            np.random.seed(self.random_state)
        self.tree = self.__grow_tree(X, y, num_bins=self.num_bins, depth=0, party_idx=self.party_idx)

    def __distribute_privacy_budged(self):
        self.leaf_epsilon = self.privacy_budget * self.leaf_privacy_proportion
        self.node_epsilon = self.privacy_budget * (1 - self.leaf_privacy_proportion) / self.max_depth

    def __grow_tree(self, X, y, num_bins, depth=0, saved_budged=0, party_idx=[]):
        # Check termination conditions to potentially create a leaf
        if (depth >= self.max_depth):
            return self.__create_leaf(X, y)
        
        #if y.size <= self.min_samples_leaf:
            #return self.__create_leaf(X, y)
        
        if self.save_budget:
            best_entropy, best_feature, best_threshold_index, saved_budged = self.__best_split_save_budged(X, y, saved_budged=saved_budged, party_idx=party_idx)
            bins = np.zeros((num_bins+1, X.shape[1]))
            for feature in range(X.shape[1]):
                bins[:, feature] = np.linspace(np.min(X[:, feature]), np.max(X[:, feature]), self.num_bins+1, endpoint=True)
        
        if self.save_budget == False:
            saved_budged = 0
            # bin the features separated by label
            X_0 = X[y == 0]
            X_1 = X[y == 1]
            #if X_0.size == 0 or X_1.size == 0:
                #return self.__create_leaf(X, y)
            # calculate binned X_0 and X_1
            bins = np.zeros((num_bins+1, X.shape[1]))
            if self.use_quantiles:
                for i in range(X.shape[1]):
                    quantile_probability = np.linspace(0, 1, self.num_bins+1, endpoint=True)
                    bins[:, i] = np.quantile(X[:, i], quantile_probability)
                binner = PrivateBinner(bins=bins, privacy_budget=self.node_epsilon, diff_privacy=self.diff_privacy)
            else:
                for i in range(X.shape[1]):
                    bins[:, i] = np.linspace(np.min(X[:, i]), np.max(X[:, i]), self.num_bins+1, endpoint=True)
                binner = PrivateBinner(bins = bins, privacy_budget=self.node_epsilon, diff_privacy=self.diff_privacy)
            X0_binned = binner.fit_transform(X_0)
            X1_binned = binner.fit_transform(X_1)
            # Create a leaf if we have less than min_samples_leaf
            # As number of samples is noisy on each feature, we check condition on all features
            n_samples = np.sum(X0_binned, axis=0) + np.sum(X1_binned, axis=0)
            if np.all(n_samples <= self.min_samples_leaf):
                return self.__create_leaf(X, y)
            
            # Create a leaf if we have only samples from one class
            # As number of samples is noisy on each feature, we check condition on all features
            n_samples_0 = np.sum(X0_binned, axis=0)
            n_samples_1 = np.sum(X1_binned, axis=0)
            if np.all(n_samples_0 <= 0) or np.all(n_samples_1 <= 0):
                return self.__create_leaf(X, y)
            
            # Find best split
            best_entropy, best_feature, best_threshold_index = self.__best_split(X0_binned, X1_binned, self.num_bins)

        # If didn't find a split that reduces entropy, make a leaf node
        if best_entropy == np.inf:
            return self.__create_leaf(X, y, saved_budged=saved_budged)
        
        # Otherwise divide space to create sub-trees
        # Find the value in the original data that corresponds to the threshold
        # (we need to do this because we binned the data)
        best_threshold = bins[best_threshold_index, best_feature]
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = X[:, best_feature] >= best_threshold

        # Update party indexes
        party_idx_left = []
        party_idx_right = []
        n_left = left_idxs.sum()
        n_right = right_idxs.sum()
        counter_left = 0
        counter_right = 0
        for party in range(len(self.party_idx)):
            # count how many go into the party
            party_left_new = np.intersect1d(party_idx[party], np.where(left_idxs)[0])
            n_party = len(party_left_new)
            # create new indexes from counter to n_party + counter
            new_idx_left = np.arange(counter_left, counter_left + n_party)
            counter_left = counter_left + n_party
            # append new indexes
            party_idx_left.append(new_idx_left)

            # repeat for right indexes
            party_right_new = np.intersect1d(party_idx[party], np.where(right_idxs)[0])
            n_party = len(party_right_new)
            new_idx_right = np.arange(counter_right, counter_right + n_party)
            counter_right = counter_right + n_party
            party_idx_right.append(new_idx_right)



        if int(left_idxs.sum() * self.num_bins) == 0 or int(right_idxs.sum() * self.num_bins) == 0:
            return self.__create_leaf(X, y)
    
        # Grow subtrees
        left_tree = self.__grow_tree(X=X[left_idxs], 
                                     y=y[left_idxs], 
                                     num_bins=self.num_bins, 
                                     depth=depth+1,
                                     saved_budged=saved_budged,
                                     party_idx=party_idx_left)
        right_tree = self.__grow_tree(X=X[right_idxs], 
                                      y=y[right_idxs], 
                                      num_bins=self.num_bins, 
                                      depth=depth+1,
                                      saved_budged=saved_budged, 
                                      party_idx=party_idx_right)

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
        
