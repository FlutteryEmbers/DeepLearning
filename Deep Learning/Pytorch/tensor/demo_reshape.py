import torch

print('==================================')
a = torch.rand(2, 3)
print(a)
out = torch.reshape(a, (3, 2))
print(out)

print('==================================')
print(torch.t(out))
print(torch.transpose(out, 0, 1))

print('==================================')
a = torch.rand(1, 2, 3)
out = torch.transpose(a, 0, 1)
print(out.shape)
print(out)

print('==================================')
out = torch.squeeze(a)
print(out)
print(out.shape)

print('==================================')
out = torch.unsqueeze(a, -1)
print(out)
print(out.shape)

print('==================================')
out = torch.unbind(a, dim=2)
print(a.shape)
print(out)

print('==================================')
print(a)
print(torch.flip(a, dims=[1, 2])) ##先对第一个维度翻转，再对第二个维度翻转

print('==================================')
print(a)
out = torch.rot90(a) ##逆时针旋转90度， -1顺时针旋转
print(out)
print(out.shape)