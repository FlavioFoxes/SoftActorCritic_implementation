import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import flatten, sigmoid
import torch.optim as optim


# Critic network: it is a Q-network. 
# It gives how much the chosen action, given the state, is good.
# Specifics:
#   - input:    the state and the action
#   - returns:  value which represents the goodness of 
#               the chosen action in the given state.
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, layers_dim=256):
        super(CriticNetwork, self).__init__()

        self.linear1 = nn.Linear(in_features = state_dim[0]+action_dim[0], out_features = layers_dim)
        self.drop1 = nn.Dropout(p=0.6)
        self.linear2 = nn.Linear(in_features = layers_dim, out_features = layers_dim)
        self.drop2 = nn.Dropout(p=0.6)
        self.linear3 = nn.Linear(in_features = layers_dim, out_features = 1)


    def forward(self, state, action):
        # Concatenate the state and the action in the input along dimension 1,
        # becuase the state and the action comes from the sampling 
        # in the replay buffer, which means they are tensor of type 
        #       (batch size, state dim)     and     (batch size, action dim) 
        # In this way input will be 
        #       (batch size, state dim + action dim)
        input = torch.cat([state, action], dim=1)

        x = self.linear1(input)
        x = F.relu(x)
        # x = self.drop1(x)

        x = self.linear2(x)
        x = F.relu(x)
        # x = self.drop2(x)

        x = self.linear3(x)

        return x

        
# For TESTING
# It seems to work. The doubt is on the dimensions of the state and of the action,
# if they can be captures through state.shape[0]. 
# I don't know if they are monodimensional tensor or not.
# A solution could be flattenize it outside the forward function and pass directly
# the total dimension of the input
if __name__=="__main__":
    state = torch.randn(2,3)
    action = torch.ra.ndn(2,1)
    print("state:   ", state.shape)
    print("action:  ", action)
    print("concat:  ", torch.cat([state, action], dim=1))
    critic = CriticNetwork(state_dim=state.shape, action_dim=action.shape[0])

    # output = critic(state, action)
    # print("output:  ", output)