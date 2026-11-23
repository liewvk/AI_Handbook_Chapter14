import numpy as np

features = np.array([2.0, 3.0])
weights = np.array([0.5, 0.8])
bias = 0.2

weighted_sum = np.dot(features, weights) + bias
activation = max(0.0, weighted_sum)
print(f"Weighted sum: {weighted_sum:.1f}")
print(f"ReLU output: {activation:.1f}")

