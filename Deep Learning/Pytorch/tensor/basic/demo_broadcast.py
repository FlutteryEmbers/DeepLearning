import torch
 ## 广播：自动补齐维度，满足右对齐（除非不一样的是1）
a = torch.rand(2, 3)
b = torch.rand(3)
c = a + b
print(a)
print(b)
print(c)
print(c.shape)

print('====================')
a = torch.rand(2, 1, 1, 3)
b = torch.rand(4, 2, 3)
# 2 * 4* 2* 3
c = a + b
print(a)
print(b)
print(c)
print(c.shape)