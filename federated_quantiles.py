import numpy as np

class FederatedPrivateQuantiles:
    def __init__(self, q, x, n):
        self.q = q # vector of quantile probbilities to compute
        #self.K = K # number of clients
        self.x = x # vector of data points
        self.n = n # total number of data points
        self.z = [0] * len(q) # vector of quantiles

    def middlepoint(self, x_min, x_max, q_prime):
        mid = (x_min + x_max) / 2
        left = np.count_nonzero(self.x < mid)
        a = left / self.n

        if len(q_prime) == 1:
            right = np.count_nonzero(self.x >= mid)  # Replace with your value or calculation
            b = 1 - right / self.n

            if a <= q_prime[0] <= b:
                self.z[np.where(q == q_prime[0])[0][0]] = mid
                self.z
                return

            if q_prime[0] <= a:
                self.middlepoint(x_min, mid, q_prime)
                return

            if q_prime[0] >= b:
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
q = np.array([0.1, 0.2, 0.3, 0.5, 0.66, 0.8, 0.95])
# create x vector with numbers from 1 to 100
x = np.array([i for i in range(1, 101)])
n = x.shape[0]

quantile_calculator = FederatedPrivateQuantiles(q, x, n)
result = quantile_calculator.compute_quantiles()
print(result)