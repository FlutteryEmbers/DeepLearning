import torch

a = torch.rand(2, 3)
b = torch.rand(2, 3)

print(a)
print(b)

print(torch.equal(a, b)) ## 返回单个值

## 以下返回的逐元素比较的结果
print(torch.eq(a, b)) 
print(torch.ge(a, b))
print(torch.le(a, b))
print(torch.gt(a, b))
print(torch.lt(a, b))
print(torch.ne(a, b))

print('Part 1')
a = torch.tensor([[1, 4, 4, 3, 5], 
                    [2, 3, 1, 3, 5]])
print(a.shape)
print(torch.sort(a, dim=0, descending=True)) ## 返回排序结果以及对应的Indexs
print(torch.sort(a, dim=1, descending=True))

## topk
print('Part 2')
a = torch.tensor([[2, 4, 3, 1, 5], [2, 3, 5, 1, 4]])
print(a.shape)
print(torch.topk(a, k=2, dim=0))

print('Part 3')
a = torch.tensor([[2, 4, 3, 1, 5], [2, 3, 5, 1, 4]])
print(torch.kthvalue(a, k=2, dim=0))
print(torch.kthvalue(a, k=2, dim=1))

print('Part 4')
a = torch.rand(2, 3)
print(a)
print(torch.isfinite(a))
print(torch.isfinite(a/0))
print(torch.isinf(a/0))
print(torch.isnan(a))

print('Part 5')
import numpy as np
a = torch.tensor([1, 2, np.nan])
print(torch.isnan(a))


