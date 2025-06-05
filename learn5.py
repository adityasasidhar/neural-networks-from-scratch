import torch
import torch.nn as nn
import torch.nn.functional as F

# Same input and class (not one-hot)
x_t = torch.tensor([1.0, 2.0], requires_grad=False).unsqueeze(0)
y_t = torch.tensor([0])  # class index

# Define model with fixed weights
model = nn.Sequential(
    nn.Linear(2, 4),   # W1
    nn.ReLU(),
    nn.Linear(4, 3)    # W2
)

# Set weights
with torch.no_grad():
    model[0].weight.copy_(torch.tensor([[0.1, 0.5],
                                        [0.2, 0.6],
                                        [0.3, 0.7],
                                        [0.4, 0.8]]))
    model[0].bias.zero_()
    model[2].weight.copy_(torch.tensor([[0.2, 0.1, 0.0, -0.2],
                                        [0.1, -0.3, 0.5, 0.1],
                                        [-0.1, 0.2, -0.4, 0.3]]))
    model[2].bias.zero_()

# Forward + Backward
criterion = nn.CrossEntropyLoss()
output = model(x_t)
loss = criterion(output, y_t)
loss.backward()

# Print autograd gradients
print("PyTorch dW1:\n", model[0].weight.grad.T)  # transpose to match shape
print("PyTorch dW2:\n", model[2].weight.grad.T)
