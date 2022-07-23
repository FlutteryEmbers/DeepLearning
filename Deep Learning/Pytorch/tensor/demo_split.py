import torch
a = torch.rand((3, 4))
out1, out2 = torch.chunk(a, 2, dim=0)
print(a)
print(out1, out1.shape)
print(out2, out2.shape)

out = torch.split(a, 2, dim=1)
print(out)

print('=======================================')
a = torch.rand((10, 4))
out = torch.split(a, 3, dim=0)
for t in out:
    print(t, t.shape)

print('=======================================')
a = torch.rand((10, 4))
out = torch.split(a, [1, 3, 6], dim=0)
for t in out:
    print(t, t.shape)
