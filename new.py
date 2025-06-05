import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load MNIST from OpenML
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data.astype(np.float32)
y = mnist.target.astype(np.int32)

# Normalize input to [0, 1]
X /= 255.0

# One-hot encode the labels (10 classes)
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))  # y is already a NumPy array

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

input_size = 784
hidden_size = 128
output_size = 10


def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # stability trick
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)  # He initialization
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    y_pred_clipped = np.clip(y_pred, 1e-12, 1. - 1e-12)
    loss = -np.sum(y_true * np.log(y_pred_clipped)) / m
    return loss

def back_propagation(X, y_true, Z1, A1, Z2, A2, W2):
    m = X.shape[0]

    dZ2 = A2 - y_true
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2

# Initialize weights
W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)

# Hyperparameters
epochs = 20
batch_size = 64
learning_rate = 0.1

num_batches = X_train.shape[0] // batch_size

for epoch in range(epochs):
    permutation = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[permutation]
    y_train_shuffled = y_train[permutation]

    epoch_loss = 0
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        X_batch = X_train_shuffled[start:end]
        y_batch = y_train_shuffled[start:end]

        Z1, A1, Z2, A2 = forward_propagation(X_batch, W1, b1, W2, b2)
        loss = compute_loss(y_batch, A2)
        epoch_loss += loss

        dW1, db1, dW2, db2 = back_propagation(X_batch, y_batch, Z1, A1, Z2, A2, W2)

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / num_batches:.4f}")

def predict(X):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    return np.argmax(A2, axis=1)

def accuracy(y_true, y_pred):
    y_true_labels = np.argmax(y_true, axis=1)
    return np.mean(y_true_labels == y_pred)

# Predict and evaluate
y_test_pred = predict(X_test)
acc = accuracy(y_test, y_test_pred)
print(f"Test Accuracy: {acc * 100:.2f}%")
