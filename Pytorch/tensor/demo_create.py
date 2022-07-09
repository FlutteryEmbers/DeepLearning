import torch as T

a = T.Tensor(2, 3)
print(a)
print(a.type())

## Special Tensor
a = T.ones(2, 2)
print(a)
print(a.type())

a = T.eye(2, 2)
print(a)
print(a.type())

a = T.zeros(2, 2)
print(a)
print(a.type())

b = T.Tensor(2, 3)
a = T.zeros_like(b)
print(a)
print(a.type())

a = T.ones_like(b)
print(a)
print(a.type())

## Random
a = T.rand(2,2)
print(a)
print(a.type())

a = T.normal(mean=0.0, std=T.rand(5))
print(a)
print(a.type())

a = T.normal(mean=T.rand(5), std=T.rand(5))
print(a)
print(a.type())

a = T.Tensor(2, 2).uniform_(-1, 1)
print(a)
print(a.type())

## Sequence
a = T.arange(0, 10, 1) ## Donesn't include last elements
print(a)
print(a.type())

a = T.linspace(2, 10, 4) ## 拿到等间隔的n个数字，等差数列，打乱数列
print(a)
print(a.type())

a = T.randperm(10)
print(a)
print(a.type())

###############################
import numpy as np
a = np.array([[1, 2], [2, 3]])
print(a)