import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self) -> None:
        super(Actor, self).__init__()