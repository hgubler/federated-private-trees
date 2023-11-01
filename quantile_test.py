import sys
from federated_private_quantiles import FederatedPrivateQuantiles
import numpy as np
from matplotlib import pyplot as plt

def simulation_plot(distributions, names, max_depths, q):
    for x in distributions:
        diffs = []
        for max_depth in max_depths:
            quantile_calculator = FederatedPrivateQuantiles(q, x, max_depth=max_depth)
            quantile_calculator.compute_quantiles()
            empirical_quantiles = np.quantile(x, q)
            diff = np.sum(np.abs(empirical_quantiles - quantile_calculator.z))
            diffs.append(diff)
        # plot the difference between compuded quantiles and empirical quantiles with max_depth on x-axis
        plt.plot(max_depths, diffs, '-')
    # add red dashed line at 0 
    plt.plot(max_depths, [0 for i in range(len(max_depths))], '--', color='red')
    plt.title("Sum of differences between computed quantiles and empirical quantiles")
    plt.xlabel("max_depth")
    plt.ylabel("sum of differences")
    plt.suptitle("n = " + str(n) + ", num_quantiles = " + str(num_quantiles))
    plt.legend(names)
    plt.savefig("plots/quantile_errors.png")

# Example usage:
n = 1000
num_quantiles = 10
q = np.array([i / n for i in range(0, n, int(n / num_quantiles))])
x_poisson = np.random.poisson(10, n)
x_gaussian = np.random.normal(0, 1, n)
x_laplace = np.random.laplace(0, 1, n)
x_cauchy = np.random.standard_cauchy(n)
x_logistic = np.random.logistic(0, 1, n)
distributions = [x_poisson, x_gaussian, x_laplace, x_cauchy, x_logistic]
# Standardize all datasets
for i in range(len(distributions)):
    distributions[i] = (distributions[i] - np.mean(distributions[i])) / np.std(distributions[i])
names = ["poisson", "gaussian", "laplace", "cauchy", "logistic"]
max_depths = [i for i in range(1, 20)]
sys.setrecursionlimit(100 + 10)
simulation_plot(distributions, names, max_depths, q)
plt.show()



