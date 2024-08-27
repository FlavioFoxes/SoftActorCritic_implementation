import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import flatten, sigmoid
import torch.optim as optim

# TODO: check if this network takes in input the state or something else
class AlphaNetwork(nn.Module):
    
    def __init__(self, state_dim, layers_dim=256):
        super(AlphaNetwork, self).__init__()

        self.linear1 = nn.Linear(in_features = state_dim, out_features = layers_dim)
        self.drop1 = nn.Dropout(p=0.6)
        self.linear2 = nn.Linear(in_features = layers_dim, out_features = layers_dim)
        self.drop2 = nn.Dropout(p=0.6)
        self.linear3 = nn.Linear(in_features = layers_dim, out_features = 1)

    def forward(self, input):
        x = self.linear1(input)
        x = F.relu(x)
        x = self.drop1(x)

        x = self.linear2(x)
        x = F.relu(x)
        x = self.drop2(x)

        x = self.linear3(x)

        return x
