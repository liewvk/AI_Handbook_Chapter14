import numpy as np

values = np.array([-2.0, 0.0, 2.0])
relu = np.maximum(0, values)
sigmoid = 1 / (1 + np.exp(-values))
scores = np.array([1.0, 2.0, 3.0])
shifted = scores - scores.max()
softmax = np.exp(shifted) / np.exp(shifted).sum()

print("ReLU:", relu.tolist())
print("Sigmoid:", np.round(sigmoid, 3).tolist())
print("Softmax:", np.round(softmax, 3).tolist())
print("Sum:", round(softmax.sum(), 3))

