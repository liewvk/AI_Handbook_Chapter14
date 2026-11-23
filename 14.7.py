import torch

w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
target = torch.tensor(10.0)
prediction = w * x
loss = (prediction - target) ** 2
loss.backward()

print("Prediction:", prediction.item())
print("Loss:", loss.item())
print("Gradient:", w.grad.item())

