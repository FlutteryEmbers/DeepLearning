import numpy as np
import torch

N = 5
QUANTS          = np.linspace(0.0, 1.0, N + 1)[1:]
# print(QUANTS)
# print(np.linspace(0.0, 1.0, N + 1)[:-1])
QUANTS_TARGET   = torch.FloatTensor((np.linspace(0.0, 1.0, N + 1)[:-1] + QUANTS)/2)
# print(QUANTS_TARGET)


a = torch.tensor([0, 100, 2, 4, 5, 6], dtype=torch.float)
a = a.reshape(-1, 2, 3)
print(a.mean(dim=2))
print(a.mean(dim=2).argmax(dim=1))
print(a*0.5)
# indices = np.arange(32)
# print(indices)


