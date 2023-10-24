import numpy as np

class FederatedPrivateQuantiles:
    def __init__(self, q, x, n):
        self.q = q # vector of quantile probbilities to compute
        #self.K = K # number of clients
        self.x = x # vector of data points
        self.n = n # total number of data points
        self.z = [0] * len(q) # vector of quantiles
        self.left_points = []

    def middlepoint(self, x_min, x_max, q_prime):
        if len(q_prime) == 0:
            return
        if q_prime[0] == 0:
            self.z[0] = self.x.min()
            q_prime = q_prime[1:]
        if q_prime[-1] == 1:
            self.z[-1] = self.x.max()
            q_prime = q_prime[:-1]
        mid = (x_min + x_max) / 2
        left = np.count_nonzero(self.x < mid)
        self.left_points.append(mid)
        a = left / self.n

        if len(q_prime) == 1:
            if a == q_prime[0]:
                self.z[np.where(q == q_prime[0])[0][0]] = mid
                return

            if q_prime[0] < a:
                self.middlepoint(x_min, mid, q_prime)
                return

            if q_prime[0] >= a:
                self.middlepoint(mid, x_max, q_prime)
                return

        midreach = False
        midind = -1
        while not midreach:
            midind += 1
            if midind == len(q_prime):
                break
            midreach = q_prime[midind] >= a 

        self.middlepoint(x_min, mid, q_prime[:midind])
        self.middlepoint(mid, x_max, q_prime[midind:])
        return

    def compute_quantiles(self):
        self.middlepoint(self.x.min(), self.x.max(), self.q)
        return self.z

# Example usage:
q = np.array([0, 0.1, 0.2, 0.3, 0.5, 0.66, 0.8, 0.95, 1])
# create x vector with numbers from 1 to 100
x = np.array([i for i in range(1, 101)])
n = x.shape[0]

quantile_calculator = FederatedPrivateQuantiles(q, x, n)
result = quantile_calculator.compute_quantiles()
print(result)
print(quantile_calculator.left_points)