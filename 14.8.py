import torch
from torch import nn

torch.manual_seed(42)
X = torch.tensor([[0., 0.], [0., 1.],
                  [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

model = nn.Sequential(
    nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1)
)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(1000):
    logits = model(X)
    loss = criterion(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    probabilities = torch.sigmoid(model(X)).flatten()
    labels = (probabilities >= 0.5).int().tolist()
print("Predicted labels:", labels)
print("Final loss:", round(loss.item(), 4))

