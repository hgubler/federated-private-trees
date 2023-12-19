import numpy as np

class SimulateData:
    def __init__(self, n_features, mean_1, mean_2, rho_1, rho_2, n_samples, seed):
        self.n_features = n_features
        self.mean_1 = mean_1
        self.mean_2 = mean_2
        self.rho_1 = rho_1
        self.rho_2 = rho_2
        self.n_samples = n_samples
        self.seed = seed
        self.cov_1 = np.zeros((self.n_features, self.n_features))
        self.cov_2 = np.zeros((self.n_features, self.n_features))

    def __create_covariance_matrix(self):
        for i in range(self.n_features):
            for j in range(self.n_features):
                self.cov_1[i, j] = self.rho_1 ** abs(i - j)
                self.cov_2[i, j] = self.rho_2 ** abs(i - j)
        return 
    
    def generate_data(self):
        self.__create_covariance_matrix()
        np.random.seed(self.seed)
        X_1 = np.random.multivariate_normal(self.mean_1, self.cov_1, self.n_samples)
        X_2 = np.random.multivariate_normal(self.mean_2, self.cov_2, self.n_samples)
        X = np.concatenate((X_1, X_2))
        y = np.concatenate((np.zeros(self.n_samples), np.ones(self.n_samples))).astype(int)
        # shuffle the data
        np.random.seed(self.seed)
        np.random.shuffle(X)
        np.random.seed(self.seed)   
        np.random.shuffle(y)
        return X, y