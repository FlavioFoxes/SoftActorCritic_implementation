import numpy as np
import torch
from torch import nn
# from alpha import AlphaNetwork
from networks.actor import ActorNetwork
from networks.critic import CriticNetwork
from replay_buffer import ReplayBuffer
import utils.utils as utils
import copy

import gymnasium as gym


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)  # Inizializzazione gaussiana
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# IMPORTANT:
# remember to make everything compatible with GYM environment,
# so the states, actions, observations, ...
class SAC():
    # TODO: check from stable-baselines (or openAI) all parameters I need, 
    #       and insert them
    def __init__(self, environment=None, lr = 3e-4, buffer_size = 1000000, batch_size = 2, tau = 0.005, gamma = 0.99,
                 gradient_steps = 1, ent_coef = 0.1) -> None:


        # TODO: add all the parameters to all variables
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.replay_buffer = ReplayBuffer(max_size=buffer_size, state_dim=environment.observation_space.shape[0], action_dim=environment.action_space.shape[0])
        self.q1_network = CriticNetwork(state_dim=environment.observation_space.shape[0], action_dim=environment.action_space.shape[0])
        self.q2_network = CriticNetwork(state_dim=environment.observation_space.shape[0], action_dim=environment.action_space.shape[0])
        self.policy = ActorNetwork(state_dim=environment.observation_space.shape[0], max_actions_values=2.0)
        # self.alpha = AlphaNetwork()
        self.alpha = ent_coef
        self.env = environment

        # Parameters of original SAC (from stable-baselines) I need
        # to make them compatible
        self.lr = lr
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.gradient_steps = gradient_steps



    # Initialize the weigths of the given network. 
    # NOTE: q1_network_target has the same initial parameters of q1_network,
    #       q2_network_target has the same initial parameters of q2_network

    # Initialize all parameters of all networks
    def initialize_networks_parameters(self):
        self.q1_network.apply(init_weights)
        self.q2_network.apply(init_weights)
        self.policy.apply(init_weights)

        # Target network must start equals to their corresponding Q-network
        self.q1_network_target = copy.deepcopy(self.q1_network)
        self.q2_network_target = copy.deepcopy(self.q2_network)
       


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
    def learn(self, total_timesteps=10, log_interval=1, tb_log_name="SAC"):
        # Move everything to device
        # utils.check_model_device(self.policy)
        
        # 1-2:  Initialize parameters of the networks
        self.initialize_networks_parameters()
        # utils.print_model_parameters(self.policy)

        # # NOTE:
        # # Observations and actions are tuples. In our environments (Classic Control)
        # # the tuples are composed by just one element ( ex: (3,) )
        # print("obs shape:   ", self.env.observation_space.shape)
        # print("action shape:   ", self.env.action_space.shape)

        for i in range(total_timesteps):
            # Reset the environment because it's a new episode
            observation, _ = self.env.reset()
            done = False
            k = 0
            # While the episode is not finished
            while not done:
                # 4:    Save the state as a tensor (and send to device because the networks are there)
                state = torch.tensor(observation, dtype=torch.float).to(self.device)
                # print("state:   ", state)

                # 4:    Sample an action through Actor network
                # It returns a tensor. The env.step want a numpy array
                action,_ = self.policy.sample_action_logprob(state, reparam_trick = False)
                # NOTE: Tensor is on device (that can be also cuda), to convert it to numpy array
                #       it needs it's on cpu and requires_grad=False, so we need to detach it
                #       (https://stackoverflow.com/questions/49768306/pytorch-tensor-to-numpy-array)
                action = action.to('cpu').detach().numpy()
                # print("action:  ", action)

                # 5-6:  Make a step in the environment, and observe (next state, reward, done)
                next_state, reward, done, _, _ = self.env.step(action)
                # print("next state:  ", next_state)
                # print("reward:  ", reward)
                # 7:    Store the transition in the replay buffer
                self.replay_buffer.store_transition(state, action, reward, next_state, done)
                
                # 9:    If the episode (after action have been applied) is not finished
                if not done and k >= 1:
                    # 10:   For the number of update steps
                    for j in range(self.gradient_steps):
                        # 11:   Sample randomly a batch of transitions   
                        # TODO: these are all numpy arrays. In the networks they must be passed
                        # as tensors. Check if computations can be done in parallel, through 
                        # many batches
                        states_batch, actions_batch, next_states_batch, rewards_batch, dones_batch = self.replay_buffer.sample_from_buffer(self.batch_size)
                        next_states = torch.tensor(next_states_batch, dtype=torch.float)
                        # print("next states:     ", next_states)
                        # 12: Compute values of target networks

                        # TODO: Actor must take in input a tensor representing one state.
                        # Check if I can compute everything for the all states together
                        actions, log_prob = self.policy.sample_action_logprob(next_states, reparam_trick=False)
                k += 1
                # done = True # just for test




if __name__=="__main__":
    # We are going to test the algorithm on Pendulum-v1,
    # because it is the most similar environment to RoboCup-v1.
    # The observation space is an array of elements, while 
    # the action space is of type Box (continuous space)
    env = gym.make('Pendulum-v1')

    # NOTE:
    # Observations and actions are tuples. In our environments (Classic Control)
    # the tuples are composed by just one element ( ex: (3,) )
    print("obs shape:   ", env.observation_space.shape)
    print("action shape:   ", env.action_space.shape)

    model = SAC(buffer_size=1000, environment=env)

    model.learn()
        