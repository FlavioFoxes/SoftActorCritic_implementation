import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch import flatten, sigmoid
import torch.optim as optim


# Actor network: it represents the policy, so how the action is chosen given the state
# Specifics:
#   - input:    the state
#   - returns:  mean and covariance of the normal distribution, through which
#               we can sample the action
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, max_actions_values, layers_dim = 256):
        super(ActorNetwork, self).__init__()
        
        # Layers of the network
        self.linear1 = nn.Linear(in_features = state_dim, out_features = layers_dim, dtype=torch.float)
        self.drop1 = nn.Dropout(p=0.6)
        self.linear2 = nn.Linear(in_features = layers_dim, out_features = layers_dim, dtype=torch.float)
        self.drop2 = nn.Dropout(p=0.6)
        self.linear_mean = nn.Linear(in_features = layers_dim, out_features = 1, dtype=torch.float)
        self.linear_variance = nn.Linear(in_features = layers_dim, out_features = 1, dtype=torch.float)
        
        # For each action, represent the max value it can have.
        # This is done because in the sampling function, we apply
        # tanh function to make it bounded between (-1,1), so then
        # it rescale according to own purposes
        # It is a tensor
        self.max_actions_values = max_actions_values

    def forward(self, state):
        input = state

        x = self.linear1(input)
        x = F.relu(x)
        x = self.drop1(x)

        x = self.linear2(x)
        x = F.relu(x)
        x = self.drop2(x)

        mean = self.linear_mean(x)
        
        # Variance is clipped in a positive interval, to avoid
        # the distribution is arbitrary large 
        variance = self.linear_variance(x)
        variance = variance.clamp(min = 1e-6, max = 1)

        return mean, variance

    # TODO: check this function
    # This function must take as argument the state and
    # it samples the action to apply in that state
    # NOTE: state is a tensor
    def sample_action_logprob(self, state, reparam_trick = True):
        # Compute mean and variance to create the Normal distribution
        mean, variance = self.forward(state)
        distributions = Normal(mean, variance)

        # If reparameterization trick is TRUE,
        # we sample from the distribution using added noise
        if(reparam_trick):
            samples = distributions.rsample()
        else:
            samples = distributions.sample()

        # Usage of tanh allows to make the samples bounded.
        # Then they have to be multiplied by the max values the actions can take
        action = F.tanh(samples)*self.max_actions_values

        # TODO: check the summation
        # Log probabilities used in the update of networks weights
        summation = (distributions.log_prob(1 - action.pow(2)))#.sum(1, keepdim=True)
        # print(distributions.log_prob(samples))
        log_probs = distributions.log_prob(samples) - summation

        # Return the action and the log prob
        return action, log_probs


# For TESTING
# TODO: check if state.shape[0] is correct to take the dimension of 
#       the state or not
if __name__=="__main__":
    state = torch.randn(3)
    print("state:      ", state)
    actor = ActorNetwork(state_dim=state.shape[0], max_actions_values=torch.randn(3))

    mean, variance = actor(state)
    print("mean:        ", mean)
    print("variance:    ", variance)



    