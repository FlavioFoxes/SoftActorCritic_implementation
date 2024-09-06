import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
# from alpha import AlphaNetwork
from actor import ActorNetwork
from critic import CriticNetwork
from replay_buffer import ReplayBuffer
# import utils.utils as utils
import copy
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import os
import random

POLICY_DIR = '/home/flavio/Scrivania/SAC_nuovo/trained_models/policy.pth'


class SAC():
    def __init__(self, environment=None, lr = 0.0003, buffer_size = 1000000, batch_size = 256, tau = 0.005, gamma = 0.99,
                 gradient_steps = 1, ent_coef = "auto", learning_starts = 5000, 
                 device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
                 tensorboard_log = '/home/flavio/Scrivania/SAC_nuovo/logs'):
        
        # TODO: add all the parameters to all variables
        self.device = device
        self.tensorboard_log = tensorboard_log
        self.env = environment

        self.replay_buffer = ReplayBuffer(max_size=buffer_size, state_dim=environment.observation_space.shape[0], action_dim=environment.action_space.shape[0])
        self.q1_network = CriticNetwork(state_dim=environment.observation_space.shape, action_dim=environment.action_space.shape)
        self.q2_network = CriticNetwork(state_dim=environment.observation_space.shape, action_dim=environment.action_space.shape)
        self.q1_network_target = CriticNetwork(state_dim=environment.observation_space.shape, action_dim=environment.action_space.shape)
        self.q2_network_target = CriticNetwork(state_dim=environment.observation_space.shape, action_dim=environment.action_space.shape)
        self.policy = ActorNetwork(state_dim=environment.observation_space.shape, max_actions_values=environment.action_space.high, device=self.device)
        # self.policy = ActorNetwork(state_dim=environment.observation_space.shape, env=environment, device=self.device)
        # self.alpha = AlphaNetwork()

        self.optimizer_critic1 = optim.Adam(self.q1_network.parameters(), lr=lr)
        self.optimizer_critic2 = optim.Adam(self.q2_network.parameters(), lr=lr)
        self.optimizer_actor = optim.Adam(list(self.policy.parameters()), lr=lr)
        self.criterion = nn.MSELoss()
        # Parameters of original SAC (from stable-baselines) I need
        # to make them compatible
        self.lr = lr
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.gradient_steps = gradient_steps
        self.learning_starts = learning_starts
        
        if ent_coef == "auto":
            self.autotune_alpha = True
            self.alpha = 0.1
            self.alpha_tensor = torch.tensor([self.alpha], requires_grad=True, dtype=torch.float32)
            self.target_entropy = -np.array(environment.action_space.shape).prod()
            self.optimizer_alpha = optim.Adam([self.alpha_tensor], lr=lr)
        else:
            self.autotune_alpha = False
            self.alpha = ent_coef

    def move_to_device(self):
        self.q1_network = self.q1_network.to(self.device)
        self.q2_network = self.q2_network.to(self.device)
        self.q1_network_target = self.q1_network_target.to(self.device)
        self.q2_network_target = self.q2_network_target.to(self.device)
        self.policy = self.policy.to(self.device)
        if self.autotune_alpha:
            self.alpha_tensor = self.alpha_tensor.to(self.device)

    def initialize_target_networks(self):
        # Target network must start equals to their corresponding Q-network
        self.q1_network_target = copy.deepcopy(self.q1_network)
        self.q2_network_target = copy.deepcopy(self.q2_network)
       

    # Return the dir name
    def get_logdir_name(self, tb_log_name):
        i = 0
        while os.path.exists(f"{os.path.join(self.tensorboard_log, tb_log_name)}_%s" % i):
            i += 1

        dir_name = tb_log_name + "_" + str(i)
        dir = os.path.join(self.tensorboard_log, dir_name)
        return dir
    
    def save(self):
        torch.save(self.policy.state_dict(), POLICY_DIR)


    def learn(self, num_episodes=250, log_interval=1, tb_log_name="SAC"):
        
        # Create Logger for debugging, with incremental filename
        dir = self.get_logdir_name(tb_log_name=tb_log_name)
        self.writer = SummaryWriter(log_dir=dir)

        self.move_to_device()
        self.initialize_target_networks()
        

        num_total_steps = 0
        for i in range(num_episodes):
            observation, _ = self.env.reset()
            done = False

            reward_record = []

            while not done:                
                state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
                if num_total_steps < self.learning_starts:
                    action = self.env.action_space.sample()
                else:
                    action, _ = self.policy.sample_action_logprob(state, reparam_trick=False)
                    action = action[0].detach().to('cpu').numpy()
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                # print(reward)
                done = terminated or truncated
                self.replay_buffer.store_transition(observation, action, reward, next_state, done)
                observation = next_state
                reward_record.append(reward)
                if num_total_steps > self.learning_starts:
                    states_sampled, actions_sampled, next_states_sampled, rewards_sampled, dones_sampled = self.replay_buffer.sample_from_buffer(self.batch_size)
                    
                    states_buffer = torch.tensor(states_sampled, dtype=torch.float32).to(self.device)
                    actions_buffer = torch.tensor(actions_sampled, dtype=torch.float32).to(self.device)
                    next_states_buffer = torch.tensor(next_states_sampled, dtype=torch.float32).to(self.device)
                    rewards_buffer = torch.tensor(rewards_sampled, dtype=torch.float32).to(self.device)
                    dones_buffer = torch.tensor(dones_sampled, dtype=torch.float32).to(self.device)

                    # -------------------------------------

                    with torch.no_grad():
                        next_actions, next_log_probs = self.policy.sample_action_logprob(next_states_buffer, reparam_trick=False)
                        q1_target_values = self.q1_network_target(next_states_buffer, next_actions)
                        q2_target_values = self.q2_network_target(next_states_buffer, next_actions)
                        q_min_target_values = torch.min(q1_target_values, q2_target_values)
                        y = rewards_buffer.unsqueeze(1) + (1 - dones_buffer.unsqueeze(1)) * self.gamma * (q_min_target_values)
                        
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

                    # -------------------------------------

                    actions, log_probs = self.policy.sample_action_logprob(states_buffer)
                    q1_values = self.q1_network(states_buffer, actions)
                    q2_values = self.q2_network(states_buffer, actions)
                    q_min_values = torch.min(q1_values, q2_values)
                    actor_loss = torch.mean((self.alpha * log_probs) - q_min_values)

                    self.optimizer_actor.zero_grad()
                    actor_loss.backward()
                    self.optimizer_actor.step()

                    # -------------------------------------

                    if self.autotune_alpha:
                        with torch.no_grad():
                            _, log_prob = self.policy.sample_action_logprob(states_buffer)
                        
                        alpha_loss = torch.mean(-self.alpha_tensor * log_prob - self.alpha_tensor * self.target_entropy)
                        # print(alpha_loss)
                        self.optimizer_alpha.zero_grad()
                        alpha_loss.backward()
                        self.optimizer_alpha.step()
                        self.alpha = self.alpha_tensor.item()

                    # -------------------------------------

                    for param, param_target in zip(self.q1_network.parameters(), self.q1_network_target.parameters()):
                        param_target.data.copy_(self.tau * param.data + (1 - self.tau) * param_target.data)

                    for param, param_target in zip(self.q2_network.parameters(), self.q2_network_target.parameters()):
                        param_target.data.copy_(self.tau * param.data + (1 - self.tau) * param_target.data)



                num_total_steps += 1

            if i % 5 == 0:
                print(f"Episode: {i}        Reward: {np.sum(reward_record)}")
                print(f"Episode: {i}        Alpha: {self.alpha}")

            if num_total_steps > self.learning_starts:
                self.writer.add_scalar("Total Reward per episode", np.sum(reward_record), i)
                self.writer.add_scalar("Q1-Loss per episode", q1_loss.item(), i)
                self.writer.add_scalar("Q2-Loss per episode", q2_loss.item(), i)
                self.writer.add_scalar("Q-Loss per episode", q_loss.item(), i)
                self.writer.add_scalar("ACTOR-Loss per episode", actor_loss.item(), i)

        self.save()

if __name__=="__main__":
    
    env = gym.make('Pendulum-v1')
    
    model = SAC(buffer_size=1000000, environment=env, ent_coef="auto")
    model.learn(num_episodes = 100)
            

    ###############################
    ###         TEST            ###
    ###############################
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # env = gym.make('Pendulum-v1', render_mode = 'human')

    # policy = ActorNetwork(state_dim=env.observation_space.shape, max_actions_values=env.action_space.high, device=device)
    # policy.load_state_dict(torch.load(POLICY_DIR, map_location=device, weights_only=True))
    
    # policy.eval()
    # policy.to(device)
    
    # state, _ = env.reset()
    # with torch.no_grad():
    #     while True:
    #             state = torch.tensor(state, dtype=torch.float).to(device)
    #             env.render()
    #             action,_ = policy(state)
    #             action = action.to('cpu').detach().numpy()
    #             next_state, reward, terminated, truncated, _ = env.step(action)
    #             state = next_state
            
        








