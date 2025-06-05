import math
import numpy as np

x = np.array([1, 2])
y = 10.0

w1 = np.random.randn(2,3)
w2 = np.random.randn(3,1)

lr = 0.01

for epoch in range(200):
    # Forward pass

    h = x @ w1
    y_hat = h @ w2
    print(y_hat)

    # Backward pass

    loss = np.mean((y_hat - y)**2)
    print(loss)

    dl_dy_hat = y_hat - y
    dl_dw2 = np.outer(h, dl_dy_hat)
    dl_dw1 = np.outer(x, (w2.flatten()*dl_dy_hat))

    w2 = w2 - lr * dl_dw2
    w1 = w1 - lr * dl_dw1

    print(f"epoch {epoch}: loss = {loss:.4f}")
