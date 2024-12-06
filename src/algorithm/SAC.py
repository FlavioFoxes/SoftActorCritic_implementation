import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import copy
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import os
from SoftActorCritic_implementation.src.networks.actor import ActorNetwork
from SoftActorCritic_implementation.src.networks.critic import CriticNetwork
from SoftActorCritic_implementation.src.algorithm.gaussian_replay_buffer import GaussianReplayBuffer
import SoftActorCritic_implementation.src.utils.utils as utils

# Path where to save policy model
POLICY_DIR = '/home/flavio/Scrivania/RoboCup/spqrnao2024/external_client/SoftActorCritic_implementation/trained_models/kick_policy.pth'

# Class that contains the whole algorithm
class SAC():
    '''
    Parameters:
        - environment:              environment
        - lr:                       learning rate for the networks trainings
        - buffer_size:              size of the replay buffer
        - batch_size:               batch size of transitions to sample
        - tau:                      update coefficient for Q-target networks
        - gamma:                    discount factor
        - gradient_steps:           how many gradient steps to do after each rollout
        - ent_coef:                 entropy regularization coefficient
        - learning_starts:          how many steps to collect transitions before starting updates
        - device:                   device
        - tensorboard_log:          path where to save logs of the trainings
    '''
    def __init__(self, environment=None, lr = 0.0003, buffer_size = 10000, batch_size = 100, tau = 0.005, gamma = 0.99,
                 gradient_steps = 1, ent_coef = "auto", learning_starts = 500, 
                 device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
                 tensorboard_log = '/home/flavio/Scrivania/Soft-Actor-Critic-implementation/logs'):
        
        self.device = device
        self.tensorboard_log = tensorboard_log
        self.env = environment

        self.replay_buffer = GaussianReplayBuffer(max_size=buffer_size, state_dim=environment.observation_space.shape[0], action_dim=environment.action_space.shape[0])
        self.q1_network = CriticNetwork(state_dim=environment.observation_space.shape, action_dim=environment.action_space.shape)
        self.q2_network = CriticNetwork(state_dim=environment.observation_space.shape, action_dim=environment.action_space.shape)
        self.q1_network_target = CriticNetwork(state_dim=environment.observation_space.shape, action_dim=environment.action_space.shape)
        self.q2_network_target = CriticNetwork(state_dim=environment.observation_space.shape, action_dim=environment.action_space.shape)
        self.policy = ActorNetwork(state_dim=environment.observation_space.shape, max_actions_values=environment.action_space.high, device=self.device)
        
        self.optimizer_critic1 = optim.Adam(self.q1_network.parameters(), lr=lr)
        self.optimizer_critic2 = optim.Adam(self.q2_network.parameters(), lr=lr)
        self.optimizer_actor = optim.Adam(list(self.policy.parameters()), lr=lr)
        self.criterion = nn.MSELoss()
        
        # Parameters of original SAC (from stable-baselines)
        self.lr = lr
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.gradient_steps = gradient_steps
        self.learning_starts = learning_starts
        
        # ent_coef can be "auto" or a number
        if ent_coef == "auto":
            self.autotune_alpha = True
            self.alpha = 0.1
            self.alpha_tensor = torch.tensor([self.alpha], requires_grad=True, dtype=torch.float32)
            self.target_entropy = -np.array(environment.action_space.shape).prod()
            self.optimizer_alpha = optim.Adam([self.alpha_tensor], lr=lr)
        else:
            self.autotune_alpha = False
            self.alpha = ent_coef

    # Move all the networks and tensors to device
    def move_to_device(self):
        self.q1_network = self.q1_network.to(self.device)
        self.q2_network = self.q2_network.to(self.device)
        self.q1_network_target = self.q1_network_target.to(self.device)
        self.q2_network_target = self.q2_network_target.to(self.device)
        self.policy = self.policy.to(self.device)
        if self.autotune_alpha:
            self.alpha_tensor = self.alpha_tensor.to(self.device)

    # Initialize Q-target networks equals to their corresponding Q-network
    def initialize_target_networks(self):
        self.q1_network_target = copy.deepcopy(self.q1_network)
        self.q2_network_target = copy.deepcopy(self.q2_network)
       
    '''
    Return the dir name
        Parameters:
            - tb_log_name:                  name of folder where to save logs for tensorboard
        Returns:
            - dir:                          (incremental) path where to save logs
    '''
    def get_logdir_name(self, tb_log_name):
        i = 0
        while os.path.exists(f"{os.path.join(self.tensorboard_log, tb_log_name)}_%s" % i):
            i += 1

        dir_name = tb_log_name + "_" + str(i)
        dir = os.path.join(self.tensorboard_log, dir_name)
        return dir
    
    # Save policy model
    def save(self):
        torch.save(self.policy.state_dict(), POLICY_DIR)

    '''
    Function that makes the entire algorithm
        Parameters:
        - num_episodes:                 number of episodes for the training
        - tb_log_name:                  name of folder where to save logs for tensorboard
    '''
    def learn(self, num_episodes=250, tb_log_name="SAC", log_interval=1):
        
        # Create Logger for debugging, with incremental filename
        dir = self.get_logdir_name(tb_log_name=tb_log_name)
        self.writer = SummaryWriter(log_dir=dir)

        # Move 
        self.move_to_device()
        self.initialize_target_networks()
        
        # DEBUG:
        # utils.check_model_device(self.policy)
        # utils.check_model_device(self.q1_network)
        # utils.check_model_device(self.q2_network)
        # utils.print_model_parameters(self.policy)

        # Total number of steps done in all the episodes
        num_total_steps = 0
        for i in range(num_episodes):
            # Reset the environment because it's a new episode
            observation, _ = self.env.reset()
            done = False

            # It contains all the rewards of the current episode
            reward_record = []

            # While not terminated or not truncated
            while not done:     
                # 4) Save the state as a tensor           
                state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # 4) Sample an action
                # If learning is not already started, sample a casual action form the action space
                if num_total_steps < self.learning_starts:
                    action = self.env.action_space.sample()
                # else sample the action through the policy
                else:
                    action, _ = self.policy.sample_action_logprob(state, reparam_trick=False)
                    # NOTE: Tensor is on device (that can be also cuda), to convert it to numpy array
                    #       it needs it's on cpu and requires_grad=False, so we need to detach it
                    #       (https://stackoverflow.com/questions/49768306/pytorch-tensor-to-numpy-array)
                    action = action[0].detach().to('cpu').numpy()
                
                # 5-6) Make a step in the environment, and observe (next state, reward, done)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # 7) Store the transition in the replay buffer
                self.replay_buffer.store_transition(observation, action, reward, next_state, done)
                observation = next_state
                reward_record.append(reward)
                
                if num_total_steps > self.learning_starts:
                    # 10) For the number of update steps
                    for j in range(self.gradient_steps):
                        # 11) Sample randomly a batch of transitions
                        states_sampled, actions_sampled, next_states_sampled, rewards_sampled, dones_sampled = self.replay_buffer.sample_from_buffer(self.batch_size)
                        
                        states_buffer = torch.tensor(states_sampled, dtype=torch.float32).to(self.device)
                        actions_buffer = torch.tensor(actions_sampled, dtype=torch.float32).to(self.device)
                        next_states_buffer = torch.tensor(next_states_sampled, dtype=torch.float32).to(self.device)
                        rewards_buffer = torch.tensor(rewards_sampled, dtype=torch.float32).to(self.device)
                        dones_buffer = torch.tensor(dones_sampled, dtype=torch.float32).to(self.device)

                        # -------------------------------------

                        with torch.no_grad():
                            # 12) Compute values of target networks (y)
                            next_actions, next_log_probs = self.policy.sample_action_logprob(next_states_buffer, reparam_trick=False)
                            q1_target_values = self.q1_network_target(next_states_buffer, next_actions)
                            q2_target_values = self.q2_network_target(next_states_buffer, next_actions)
                            q_min_target_values = torch.min(q1_target_values, q2_target_values)
                            y = rewards_buffer.unsqueeze(1) + (1 - dones_buffer.unsqueeze(1)) * self.gamma * (q_min_target_values)
                            
                        # 13) Update of Q-functions (Critics Q1 and Q2)
                        q1_output = self.q1_network(states_buffer, actions_buffer)
                        q2_output = self.q2_network(states_buffer, actions_buffer)

                        q1_loss = self.criterion(q1_output, y)
                        q2_loss = self.criterion(q2_output, y)
                        q_loss = q1_loss + q2_loss
                        # print("qloss       ", q_loss)

                        self.optimizer_critic1.zero_grad()
                        self.optimizer_critic2.zero_grad()
                        q_loss.backward()
                        self.optimizer_critic1.step()
                        self.optimizer_critic2.step()

                        # 14.1) Update of policy (Actor) through Gradient Ascent
                        actions, log_probs = self.policy.sample_action_logprob(states_buffer)
                        q1_values = self.q1_network(states_buffer, actions)
                        q2_values = self.q2_network(states_buffer, actions)
                        q_min_values = torch.min(q1_values, q2_values)
                        actor_loss = torch.mean((self.alpha * log_probs) - q_min_values)

                        self.optimizer_actor.zero_grad()
                        actor_loss.backward()
                        self.optimizer_actor.step()

                        # 14.2) Update entropy coefficient (alpha)
                        if self.autotune_alpha:
                            with torch.no_grad():
                                _, log_prob = self.policy.sample_action_logprob(states_buffer)
                            
                            alpha_loss = torch.mean(-self.alpha_tensor * log_prob - self.alpha_tensor * self.target_entropy)
                            # print(alpha_loss)
                            self.optimizer_alpha.zero_grad()
                            alpha_loss.backward()
                            self.optimizer_alpha.step()
                            self.alpha = self.alpha_tensor.item()

                        # 15) Update target networks (Q1-target and Q2-target)
                        for param, param_target in zip(self.q1_network.parameters(), self.q1_network_target.parameters()):
                            param_target.data.copy_(self.tau * param.data + (1 - self.tau) * param_target.data)

                        for param, param_target in zip(self.q2_network.parameters(), self.q2_network_target.parameters()):
                            param_target.data.copy_(self.tau * param.data + (1 - self.tau) * param_target.data)

                num_total_steps += 1
            
            # Print Reward and Alpha every 5 steps
            if i % log_interval == 0:
                print(f"Episode: {i}        Reward: {np.sum(reward_record)}")
                print(f"Episode: {i}        Alpha: {self.alpha}")

            # Logs for TensorBoard
            if num_total_steps > self.learning_starts + 1 and i%log_interval==0:
                self.writer.add_scalar("Total Reward per episode", np.sum(reward_record), i)
                self.writer.add_scalar("Q1-Loss per episode", q1_loss.item(), i)
                self.writer.add_scalar("Q2-Loss per episode", q2_loss.item(), i)
                self.writer.add_scalar("Q-Loss per episode", q_loss.item(), i)
                self.writer.add_scalar("ACTOR-Loss per episode", actor_loss.item(), i)
                self.writer.add_scalar("Alpha value", self.alpha, i)

        # Save policy
        self.save()

if __name__=="__main__":
    
    env = gym.make('Pendulum-v1')
    
    model = SAC(buffer_size=1000000, environment=env, ent_coef="auto")
    model.learn(num_episodes = 100)
            







