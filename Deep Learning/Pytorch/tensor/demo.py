import torch

a = torch.rand(2, 3) * 10
print(a)

out = torch.split(a, 2, dim=1)
print(out)