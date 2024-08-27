import numpy as np
import torch
from torch import nn
# from alpha import AlphaNetwork
from networks.actor import ActorNetwork
from networks.critic import CriticNetwork
from replay_buffer import ReplayBuffer
import utils.utils as utils

# IMPORTANT:
# remember to make everything compatible with GYM environment,
# so the states, actions, observations, ...
class SAC():
    # TODO: check from stable-baselines (or openAI) all parameters I need, 
    #       and insert them
    def __init__(self, max_size_buffer) -> None:
        # TODO: add all the parameters to all variables
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.replay_buffer = ReplayBuffer(max_size=max_size_buffer, shape=8, num_actions=2)
        self.q1_network = CriticNetwork(state_dim=3, action_dim=8)
        self.q2_network = CriticNetwork(state_dim=3, action_dim=8)
        self.q1_network_target = CriticNetwork(state_dim=3, action_dim=8)
        self.q2_network_target = CriticNetwork(state_dim=3, action_dim=8)
        self.policy = ActorNetwork(state_dim=3, max_actions_values=2.0)
        # self.alpha = AlphaNetwork()

    # Initialize the parameters of the networks. 
    # NOTE: q1_network_target has the same initial parameters of q1_network,
    #       q2_network_target has the same initial parameters of q2_network
    def initialize_parameters(m):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)  # Inizializzazione gaussiana
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)



    # Update the parameters of the networks
    def update_parameters():
        pass

    def move_to_device(self):
        self.q1_network = self.q1_network.to(self.device)
        self.q2_network = self.q2_network.to(self.device)
        self.q1_network_target = self.q1_network_target.to(self.device)
        self.q2_network_target = self.q2_network_target.to(self.device)
        self.policy = self.policy.to(self.device)
        # self.alpha = self.alpha.to(self.device)
    # Function that makes everything work.
    # Params:   - timesteps
    def learn(self, total_timesteps=1000, log_interval=1, tb_log_name="SAC"):
        # Move everything to device
        utils.check_model_device(self.policy)
        

if __name__=="__main__":
    sac = SAC(max_size_buffer=1000)
    sac.learn()
        