import sys
from federated_private_quantiles import FederatedPrivateQuantiles
import numpy as np
from matplotlib import pyplot as plt

# Example usage:
n = 1000
num_quantiles = 10
x = np.random.poisson(10, n)
q = np.array([i / len(x) for i in range(0, len(x), int(len(x) / num_quantiles))])
max_depth = n
quantile_calculator = FederatedPrivateQuantiles(q, x, max_depth=max_depth)
sys.setrecursionlimit(max_depth + 10)
quantile_calculator.compute_quantiles()
print(quantile_calculator.z)
print(quantile_calculator.left_points_counter)

empirical_quantiles = np.quantile(x, q)
# plot the difference between compuded quantiles and empirical quantiles with max_depth on x-axis
max_depths = [i for i in range(1, 100)]
diffs = []
for max_depth in max_depths:
    quantile_calculator = FederatedPrivateQuantiles(q, x, max_depth=max_depth)
    sys.setrecursionlimit(max_depth + 10)
    quantile_calculator.compute_quantiles()
    diff = np.sum(np.abs(empirical_quantiles - quantile_calculator.z))
    diffs.append(diff)

plt.plot(max_depths, diffs, 'x')
plt.title("Sum of differences between computed quantiles and empirical quantiles")
plt.xlabel("max_depth")
plt.ylabel("sum of differences")
plt.suptitle("n = " + str(n) + ", num_quantiles = " + str(num_quantiles))
plt.show()

