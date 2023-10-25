import numpy as np

class FederatedPrivateQuantiles:
    """
    A class for computing quantiles of a dataset using the federated private quantiles algorithm.

    Attributes:
        q (numpy.ndarray): A sorted vector of quantile probabilities. Need to be of the form i/n, i between 0 and n.
        x (numpy.ndarray): A vector of data points.
        n (int): The total number of data points.
        z (list): A vector of quantiles to compute.
        left_points_counter: A counter of how many datapoints where looked at during quantile computation.

    Methods:
        middlepoint: A recursive function that computes the quantiles using the federated private quantiles algorithm.
        compute_quantiles: A function that calls middlepoint to compute the quantiles and return the result.
    """
    def __init__(self, q, x):
        self.q = q # vector of quantile probbilities to compute
        #self.K = K # number of clients
        self.x = x # vector of data points
        self.n = len(x) # total number of data points
        self.z = [0] * len(q) # vector of quantiles
        self.left_points_counter = 0 # counter of how many times we compute left points 
        # (hence how many times we exchange information between clients)

    def middlepoint(self, x_min, x_max, q_prime):
        """
        A recursive function that computes the quantiles using the federated private quantiles algorithm.

        Args:
            x_min (float): The minimum value of the data points.
            x_max (float): The maximum value of the data points.
            q_prime (numpy.ndarray): A vector of quantile probabilities to compute.

        Returns:
            None
        """
        if len(q_prime) == 0: # nothing to compute in this interval
            return
        if q_prime[0] == 0: # want to compute 0 quantile, hence return minimum and remove 0 from q_prime
            self.z[0] = self.x.min()
            q_prime = q_prime[1:]
        if q_prime[-1] == 1: # want to compute 1 quantile, hence return maximum and remove 1 from q_prime
            self.z[-1] = self.x.max()
            q_prime = q_prime[:-1]
        mid = (x_min + x_max) / 2 # calculate middle point
        left = np.count_nonzero(self.x < mid) # calculate how many points are on the left side of the middle point
        self.left_points_counter += 1 # increment counter of left points computed
        a = left / self.n # calculate the quantile of the middle point

        if len(q_prime) == 1:
            if a == q_prime[0]: # if the quantile of the middle point is the one we want to compute, return it
                self.z[np.where(q == q_prime[0])[0][0]] = mid
                return


        midreach = False
        midind = -1
        while not midreach: # find the index of the first quantile in q_prime that is bigger than the quantile of the middle point
            midind += 1
            if midind == len(q_prime):
                break
            midreach = q_prime[midind] >= a 

        # search in the left interval for the quantiles
        # in q_prime that are smaller than the quantile of the middle point
        self.middlepoint(x_min, mid, q_prime[:midind])
        # search in the right interval for the quantiles
        # in q_prime that are bigger than the quantile of the middle point
        self.middlepoint(mid, x_max, q_prime[midind:])
        return

    def compute_quantiles(self):
        """
        Computes the quantiles and returns the result.

        Returns:
            None
        """
        # assert that quantiles are of the form i/n, i between 0 and n
        possible_quantiles = [i / self.n for i in range(self.n + 1)]
        assert np.all(np.isin(self.q, possible_quantiles)), "Quantiles need to be in the form i/n, where i is between 0 and n."
        # assert that quantiles are sorted
        assert np.all(np.diff(self.q) > 0), "Quantiles need to be sorted."
        self.middlepoint(self.x.min(), self.x.max(), self.q)
        return self.z

# Example usage:
# create x vector with numbers from 1 to 100
x = np.array([i for i in range(1, 101)])
n = x.shape[0]
q = np.array([i / n for i in range(0, n + 1, 8)])

quantile_calculator = FederatedPrivateQuantiles(q, x)
quantile_calculator.compute_quantiles()
print(quantile_calculator.z)
print(quantile_calculator.left_points_counter)