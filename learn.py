import math

x = 2.0
y = 1.0

w = 0.5
b = 1

lr = 0.01

z = ((x * w) + b)

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def loss(y_hat, y):
    return 0.5*((y_hat - y)**2)

for epoch in range(10):
    z = ((x * w) + b)
    y_hat = sigmoid(z)
    loss_value = loss(y_hat, y)

    print(f"Epoch {epoch}: Loss = {loss_value}")
    dl_dw = x * (y_hat - y) * sigmoid_derivative(z)
    w = w - (lr * dl_dw)
    dl_db = (y_hat - y) * sigmoid_derivative(z)
    b = b - (lr * dl_db)
    dl_db = (y_hat - y) * sigmoid_derivative(z)
    b = b - (lr * dl_db)

    print(f"Epoch {epoch+1}: w = {w}, b = {b}, loss = {loss}")
