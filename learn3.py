import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([1.0,2.0], requires_grad=False).unsqueeze(0)
y = torch.tensor([0])

model = nn.Sequential(
nn.Linear(2,4),
nn.ReLU(),
nn.Linear(4,3)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


for epoch in range(50):
    optimizer.zero_grad()

    out = model(x)
    loss = criterion(out, y)

    loss.backward()
    optimizer.step()

    if epoch % 1 == 0:
        print(f'The predicted output is {out}, Epoch {epoch}, Loss: {loss.item()}')
