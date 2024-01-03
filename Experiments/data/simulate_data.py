import numpy as np

class SimulateData:
    """
    Simulate data from a Gaussian mixture model or from a single Gaussian distribution.

    Attributes
    ----------
        n_features : int
            The number of features in the data. the default is 10.
        mean_1 : numpy array
            The mean of the first Gaussian distribution of class 0 (only relevant if mixture_model=False).
        mean_2 : numpy array
            The mean of the second Gaussian distribution of class 1 (only relevant if mixture_model=False).
        rho_1 : float
            The correlation determining the covariance matrix of class 0. The default is 0.5.
        rho_2 : float
            The correlation determining the covariance matrix of class 1. The default is 0.5.
        n_samples : int
            The number of samples to generate. The default is 100.
        n_centers : int
            The number of centers in the mixture model (only relevant if mixture_model=True). The default is 5.
        mixture_model : bool
            Whether to generate data from a mixture model or not. The default is True.
        seed : int
            The random seed to use. The default is 0.

    Methods
    -------
        generate_data():
            Generates the data.   
    """
    def __init__(self, n_features=10, mean_1=None, mean_2=None, rho_1=0.5, rho_2=0.5, n_samples=100, n_centers=5, mixture_model=True, seed=0):
        self.n_features = n_features
        self.mean_1 = mean_1
        self.mean_2 = mean_2
        self.rho_1 = rho_1
        self.rho_2 = rho_2
        self.n_samples = n_samples
        self.mixture_model = mixture_model
        self.seed = seed
        self.n_centers = n_centers
        self.cov_1 = np.zeros((self.n_features, self.n_features))
        self.cov_2 = np.zeros((self.n_features, self.n_features))

    def __create_covariance_matrix(self):
        """
        Creates the covariance matrices for the two classes.

        Returns
        -------
            None
        """
        if self.rho_1 == 0 and self.rho_2 == 0:
            self.cov_1 = np.eye(self.n_features)
            self.cov_2 = np.eye(self.n_features)
            return
        for i in range(self.n_features):
            for j in range(self.n_features):
                self.cov_1[i, j] = self.rho_1 ** abs(i - j)
                self.cov_2[i, j] = self.rho_2 ** abs(i - j)
        return 
    
    def generate_data(self):
        """
        Generates the data.

        Returns
        -------
            tuple
                The features and labels of the generated data.
        """
        if self.mixture_model:
            X, y = self.__sample_gaussian_mixture()
        if self.mixture_model == False:
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
    
    def __sample_gaussian_mixture(self):
        """
        Samples from a Gaussian mixture model.

        Returns
        -------
            tuple
                The features and labels of the generated data.
        """
        # create isotropic covariance matrices
        self.__create_covariance_matrix()
        center_probability = np.zeros(self.n_centers)
        for i in range(self.n_centers):
            center_probability[i] = i^2
        center_probability = center_probability / center_probability.sum()
        centers_0 = []
        for i in range(self.n_centers):
            centers_0.append(np.ones(self.n_features) * (i+1) / self.n_centers)
        centers_1 = []
        for i in range(self.n_centers):
            centers_1.append(np.ones(self.n_features) * -(i+1) / self.n_centers)
        # sample features from the mixture model
        X_0 = np.zeros((self.n_samples, self.n_features))
        X_1 = np.zeros((self.n_samples, self.n_features))
        for i in range(self.n_samples):
            center = np.random.choice(self.n_centers)
            X_0[i, :] = np.random.multivariate_normal(centers_0[center], self.cov_1)
            X_1[i, :] = np.random.multivariate_normal(centers_1[center], self.cov_2)
        X = np.concatenate((X_0, X_1))
        # create labels
        y = np.concatenate((np.zeros(self.n_samples), np.ones(self.n_samples))).astype(int)
        return X, y
    