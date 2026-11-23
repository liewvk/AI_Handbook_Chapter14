import numpy as np

X = np.array([[1.0, 2.0], [2.0, 1.0]])
W1 = np.array([[1.0, -1.0], [0.5, 1.0]])
b1 = np.array([0.0, 0.5])
W2 = np.array([[0.6], [-0.2]])
b2 = np.array([0.1])

hidden = np.maximum(0, X @ W1 + b1)
prediction = hidden @ W2 + b2

print("Hidden layer:\n", hidden)
print("Predictions:", prediction.ravel().tolist())
