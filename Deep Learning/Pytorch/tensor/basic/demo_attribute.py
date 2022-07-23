import torch

device = torch.device("cuda:0")

a = torch.tensor([2, 2], device=device, dtype=torch.float32)
print(a)

coord = torch.tensor([[0, 1, 2], [0, 1, 2]])
v = torch.tensor([1, 2, 3])
a = torch.sparse_coo_tensor(coord, v, (4,4)).to_dense()

print(a)

