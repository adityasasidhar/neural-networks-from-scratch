import numpy as np

# Batch of 3 input samples (each with 2 features)
X = np.array([
    [1.0, 2.0],
    [0.5, -1.0],
    [3.0, 0.2]
])  # shape (3, 2)

# One-hot encoded targets for 3 classes
Y = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])  # shape (3, 3)

# Set fixed weights
W1 = np.array([[0.1, 0.2, 0.3, 0.4],
               [0.5, 0.6, 0.7, 0.8]])  # shape (2, 4)
W2 = np.array([[0.2, 0.1, -0.1],
               [0.1, -0.3, 0.2],
               [0.0, 0.5, -0.4],
               [-0.2, 0.1, 0.3]])       # shape (4, 3)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Forward
z1 = X @ W1
h = relu(z1)
z2 = h @ W2
y_hat = softmax(z2)

loss = -np.sum(Y * np.log(y_hat + 1e-9)) / X.shape[0]

dz2 = (y_hat - Y) / X.shape[0]  # (3, 3)
dW2 = h.T @ dz2                 # (4, 3)
dh = dz2 @ W2.T                # (3, 4)
dz1 = dh * relu_deriv(z1)      # (3, 4)
dW1 = X.T @ dz1                # (2, 4)

print("Manual Batch dW1:\n", dW1)
print("Manual Batch dW2:\n", dW2)
print("Loss:", loss)
