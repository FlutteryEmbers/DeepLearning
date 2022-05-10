import torch

array = torch.tensor([[1,2,3], [4, 5, 6], [7, 8, 9], [10 , 11, 12]])
array = array.reshape(1, 3, 4)
print(array)