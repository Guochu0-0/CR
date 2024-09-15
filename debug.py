import torch

x = torch.ones([1, 3, 16, 16])

y = x.sum(-3)
print(y)