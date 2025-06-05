import numpy as np

# Input and true output
x = np.array([1.0, 2.0])
y = np.array([1, 0, 0])  # class 0

# Fixed weights for reproducibility
W1 = np.array([[0.1, 0.2, 0.3, 0.4],
               [0.5, 0.6, 0.7, 0.8]])
W2 = np.array([[0.2, 0.1, -0.1],
               [0.1, -0.3, 0.2],
               [0.0, 0.5, -0.4],
               [-0.2, 0.1, 0.3]])

def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)
def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

# Forward
z1 = x @ W1
h = relu(z1)
z2 = h @ W2
y_hat = softmax(z2)
loss = -np.sum(y * np.log(y_hat + 1e-9))

# Backward
dz2 = y_hat - y
dW2 = np.outer(h, dz2)

dh = W2 @ dz2
dz1 = dh * relu_deriv(z1)
dW1 = np.outer(x, dz1)

print("Manual dW1:\n", dW1)
print("Manual dW2:\n", dW2)
