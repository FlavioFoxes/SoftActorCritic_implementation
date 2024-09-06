import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.distributions import Normal



# Actor network: it represents the policy, so how the action is chosen given the state
class ActorNetwork(nn.Module):
    '''
    Parameters:
        -   state_dim:              shape of the observation space
        -   max_action_values:      the highest value the action can take
        -   layers_dim:             dimension of the linear layers
        -   device:                 where to do computation (cpu or cuda)
    '''
    def __init__(self, state_dim, max_actions_values, layers_dim = 256, device = None):
        super(ActorNetwork, self).__init__()
        self.device = device
        self.max_actions_values = max_actions_values
        
        # Layers of the network
        self.linear1 = nn.Linear(in_features = state_dim[0], out_features = layers_dim, dtype=torch.float32)
        self.linear2 = nn.Linear(in_features = layers_dim, out_features = layers_dim, dtype=torch.float32)
        self.linear_mean = nn.Linear(in_features = layers_dim, out_features = 1, dtype=torch.float32)
        self.linear_stddev = nn.Linear(in_features = layers_dim, out_features = 1, dtype=torch.float32)
        

    '''
        Parameters: 
        -   state:                  state (tensor) from which we want to compute the action

        Returns:
        -   mean:                   mean of the Normal distribution, through which we compute the action
        -   stddev:                 standard deviation of the Normal distribution, through which we compute the action
    '''
    def forward(self, state):
        input = state

        x = self.linear1(input)
        x = F.relu(x)
        # x = self.drop1(x)

        x = self.linear2(x)
        x = F.relu(x)
        # x = self.drop2(x)

        mean = self.linear_mean(x)
        stddev = self.linear_stddev(x)
        
        # Standard deviation is clipped in a positive interval, to avoid
        # the distribution is arbitrary large 
        stddev = stddev.clamp(min = 1e-6, max = 1)
        return mean, stddev


    '''
    This function compute the Normal distribution, sample an action from it and
    computes its log_prob

        Parameters:
        -   state:                  state (tensor) from which we want to compute the action
        -   reparam_trick:          if we want to use reparametrization trick

        Returns:
        -   actions:                sampled action
        -   log_probs
    
    NOTE: state can be also a batch of states -> action and log_probs will be a batch of actions, log_probs
    '''
    def sample_action_logprob(self, state, reparam_trick = True):
        max_action_tensor = torch.tensor(self.max_actions_values, dtype=torch.float32).to(self.device)
        
        # Compute mean and variance to create the Normal distribution
        mean, stddev = self(state)
        distributions = Normal(mean, stddev)

        # If reparameterization trick is TRUE,
        # we sample from the distribution using added noise
        samples = distributions.rsample() if reparam_trick else distributions.sample()

        # Usage of tanh allows to make the samples bounded.
        # Then they have to be multiplied by the max values the actions can take,
        # to guarantee original action scale.
        # NOTE: epsilon is necessary, to avoid cases where log(0)
        epsilon = 1e-6
        tanh_samples = torch.tanh(samples)
        action = tanh_samples * max_action_tensor

        # Log probabilities used in the update of networks weights
        summation = torch.log((1 - tanh_samples.pow(2)) * max_action_tensor + epsilon)
        log_probs = distributions.log_prob(samples) - summation
        log_probs = log_probs.sum(dim = 1, keepdim = True)
        
        # Return the action and the log prob
        return action, log_probs

if __name__ == "__main__":
    import gymnasium as gym
    from actor import ActorNetwork
    from critic import CriticNetwork
    
    env = gym.make('Pendulum-v1')
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    print(state)
    policy = ActorNetwork(state_dim = env.observation_space.shape, env = env)

    action, log_probs = policy.sample_action_logprob(state)

