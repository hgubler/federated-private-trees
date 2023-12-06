import numpy as np

class PrivateBinner:
    def __init__(self, num_bins, privacy_budget=1, diff_privacy=True, quantiles = None):
        self.num_bins = num_bins
        self.privacy_budget = privacy_budget
        self.diff_privacy = diff_privacy
        self.quantiles = quantiles

    def fit_transform(self, X):
        # Set X_binned to a matrix with number of bins as rows and number of features as columns
        X_binned = np.zeros((self.num_bins, X.shape[1]))
        epsilon = self.privacy_budget / X.shape[1]
        # if the private histogram for each feature is parralel composition
        # epsilon = self.privacy_budget
        if self.quantiles is None:
            self.bins = np.zeros((self.num_bins+1, X.shape[1]))
        if self.quantiles is not None:
            self.bins = self.quantiles
        for i in range(X.shape[1]):
            # calculate quantiles of the feature
            if X.size == 0:
                raise ValueError("Input array X is empty")
            if self.quantiles is None:
                bincounts, bins = np.histogram(X[:, i], bins=self.num_bins)
                self.bins[:, i] = bins
            if self.quantiles is not None:
                quantiles = self.quantiles[:, i]
                bincounts, bins = np.histogram(X[:, i], bins=quantiles)
            # store the bincounts in X_bins
            if self.diff_privacy:
                X_binned[:, i] = bincounts + np.random.laplace(0, 1/epsilon, size=bincounts.shape[0])
            else:
                X_binned[:, i] = bincounts
        return np.round(X_binned).astype(int)
    
# test code
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    private_binner = PrivateBinner(num_bins=10, privacy_budget=1)
    X_binned = private_binner.fit_transform(X)
    print(X_binned)
    import matplotlib.pyplot as plt
    import pandas as pd
    # convert X to pandas dataframe
    X_df = pd.DataFrame(X)
    X_df.hist()
    plt.show()