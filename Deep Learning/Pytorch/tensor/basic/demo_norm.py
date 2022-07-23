import torch

a = torch.rand(2, 1)
b = torch.rand(2, 1)

print(a)
print(b)

print('====================================')
print(torch.dist(a, b, p=1))
print(torch.dist(a, b, p=2))
print(torch.dist(a, b, p=3))

print('====================================')
print(torch.norm(a, p=1))
print(torch.norm(a, p=2))

print('====================================')
print(torch.norm(a, p='fro')) ## 核范数