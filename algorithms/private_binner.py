import numpy as np

class PrivateBinner:
    def __init__(self, num_bins, privacy_budget=1):
        self.num_bins = num_bins
        self.privacy_budget = privacy_budget

    def fit_transform(self, X):
        # Set X_binned to a matrix with number of bins as rows and number of features as columns
        X_binned = np.zeros((self.num_bins, X.shape[1]))
        epsilon = self.privacy_budget / X.shape[1]
        for i in range(X.shape[1]):
            bincounts, bins = np.histogram(X[:, i], bins=self.num_bins)
            # store the bincounts in X_bins
            X_binned[:, i] = bincounts + np.random.laplace(0, 1/epsilon, size=bincounts.shape[0])
        return X_binned
    
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