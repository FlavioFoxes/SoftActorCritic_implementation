import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

# Critic network: it is a Q-network. 
# It gives how much the chosen action is good, given the state.
'''
Specifics:
  - input:    the state and the action
  - returns:  value which represents the goodness of 
              the chosen action in the given state.
'''
class CriticNetwork(nn.Module):
    '''
    Parameters:
        -   state_dim:              shape of the observation space
        -   action_dim:             shape of the action space
        -   layers_dim:             dimension of the linear layers
    '''
    def __init__(self, state_dim, action_dim, layers_dim=256):
        super().__init__()
        self.linear1 = nn.Linear(in_features = state_dim[0]+action_dim[0], out_features = layers_dim)
        self.linear2 = nn.Linear(in_features = layers_dim, out_features = layers_dim)
        self.linear3 = nn.Linear(in_features = layers_dim, out_features = 1)

    '''
    Parameters:
        -   state:                  state (tensor) from which we want to compute the action
        -   action:                 action (tensor) applied at state 

    Returns:
        -   x:                      goodness of action applied in state        
    '''
    def forward(self, state, action):
        # State and action must be concatenated
        input = torch.cat([state, action], dim=1)

        x = self.linear1(input)
        x = F.relu(x)

        x = self.linear2(x)
        x = F.relu(x)

        x = self.linear3(x)
        return x