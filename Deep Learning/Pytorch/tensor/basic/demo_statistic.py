import torch

a = torch.rand(2, 2)
print(a)
print(a.mean(dim=0))
print(a.sum())
print(a.prod())

print('==================================')
print(torch.argmax(a, dim=0))
print(torch.argmin(a, dim=1))

print('==================================')
print(torch.median(a))
print(torch.mode(a))

print('==================================')
a = torch.rand(2, 2) * 10
print(a)
print(torch.histc(a, 5, 0, 20))

## 统计某一类别的个数进行统计
print('==================================')
a = torch.randint(0, 10, [10])
print(a)
print(torch.bincount(a))
