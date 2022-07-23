import torch
# torch.where
a = torch.rand(4, 4)
b = torch.rand(4, 4)
print('a', a)
print('b', b)
out = torch.where(a > 0.5, a, b) ## out_ij = a_ij if a_ij > 0.5 else b_ij
print('where = ')
print(out)

# torch.index_select
print('===========================================')
a = torch.rand(4, 4)
print(a)
out = torch.index_select(a, dim=0, index=torch.tensor([0, 3, 2]))
print('index_select = ')
print(out, out.shape)

# torch.gather
## 高维gather: 
## dim = 0: out[i, j, k] = input[index[i, j, k], j, k]
## dim = 1: out[i, j, k] = input[i, index[i, j, k], k]
## dim = 2: out[i, j, k] = input[i, j, index[i, j, k]]
print('===========================================')
a = torch.linspace(1, 16, 16).view(4, 4)
print(a)
out = torch.gather(a, dim=0, index=torch.tensor([[0, 1, 1, 1],
                                                 [0, 1, 2, 2],
                                                 [0, 1, 3, 3]])) ## dim=0 就是按照列索引； 和index是有区别的， 对照列上的索引值，每个值对应一个维度
print('gather = ')
print(out)

print('===========================================')
a = torch.linspace(1, 16, 16).view(4, 4)
print(a)
mask = torch.gt(a, 8)
print(mask)
out = torch.masked_select(a, mask)
print('mask = ')
print(out)

print('===========================================')
a = torch.linspace(1, 16, 16).view(4, 4)
print(a)
b = torch.take(a, index=torch.tensor([0, 15, 13, 10]))
print('take = ')
print(b)

print('===========================================')
a = torch.tensor([[0, 1, 2, 0], [2, 3, 0, 1]])
print(a)
out = torch.nonzero(a)
print('nonzero = ')
print(out)




