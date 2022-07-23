import torch

a = torch.zeros((2, 4))
b = torch.ones((2, 4))

c = torch.cat((a, b), dim=0)
print(c)

c = torch.cat((a, b), dim=1)
print(c)

a = torch.linspace(1, 6, 6).view(2, 3)
b = torch.linspace(7, 12, 6).view(2, 3)
# print(a)
# print(b)
c = torch.stack((a, b), dim=0)
print(c, c.shape)
# print(c[0, :, :]) ##输出的是A
# print(c[1, :, :]) ##输出的是B

c2 = torch.stack((a, b), dim=1)
print(c2, c2.shape)
print(c2[:, 0, :]) ##输出的是A
print(c2[:, 1, :]) ##输出的是B

