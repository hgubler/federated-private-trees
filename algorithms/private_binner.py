import numpy as np

class PrivateBinner:
    def __init__(self, bins, privacy_budget=1, diff_privacy=True):
        self.bins = bins
        self.privacy_budget = privacy_budget
        self.diff_privacy = diff_privacy

    def fit_transform(self, X):
        # Set X_binned to a matrix with number of bins as rows and number of features as columns
        num_bins = self.bins.shape[0] - 1
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_binned = np.zeros((num_bins, X.shape[1]))
        epsilon = self.privacy_budget / X.shape[1]

        for i in range(X.shape[1]):
            # calculate quantiles of the feature
            if X.size == 0:
                if self.diff_privacy:
                    X_binned[:, i] = np.random.laplace(0, 1/epsilon, size=X_binned[:, i].shape[0])
                continue

            bincounts, bins = np.histogram(X[:, i], bins=self.bins[:, i])
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
    bins = np.zeros((10+1, X.shape[1]))
    for i in range(X.shape[1]):
        bins[:, i] = np.linspace(np.min(X[:, i]), np.max(X[:, i]), 10+1, endpoint=True)
    private_binner = PrivateBinner(bins=bins, privacy_budget=1)
    X_binned = private_binner.fit_transform(X)
    print(X_binned)
    import matplotlib.pyplot as plt
    import pandas as pd
    # convert X to pandas dataframe
    X_df = pd.DataFrame(X)
    X_df.hist(bins=10)
    plt.show()