import numpy as np
import torch
from torch import nn
import torch.optim as optim
# from alpha import AlphaNetwork
from networks.actor import ActorNetwork
from networks.critic import CriticNetwork
from networks.value import ValueNetwork
from replay_buffer import ReplayBuffer
import utils.utils as utils
import copy
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import os

# Initialize the weigths of the given network. 
# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         nn.init.normal_(m.weight, mean=0.0, std=0.02)  # Inizializzazione gaussiana
#     elif isinstance(m, nn.BatchNorm2d):
#         nn.init.constant_(m.weight, 1)
#         nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.BatchNorm1d):
#         nn.init.constant_(m.weight, 1)
#         nn.init.constant_(m.bias, 0)
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, a=-0.5, b=0.5)  # Inizializzazione gaussiana
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.uniform_(m.weight, a=-0.5, b=0.5)
        nn.init.uniform_(m.bias, a=-0.5, b=0.5)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.uniform_(m.weight, a=-0.5, b=0.5)
        nn.init.uniform_(m.bias, a=-0.5, b=0.5)

# IMPORTANT:
# remember to make everything compatible with GYM environment,
# so the states, actions, observations, ...
class SAC_value():
    # TODO: check from stable-baselines (or openAI) all parameters I need, 
    #       and insert them
    def __init__(self, environment=None, lr = 0.001, buffer_size = 1000000, batch_size = 256, tau = 0.005, gamma = 0.95,
                 gradient_steps = 1, ent_coef = 0.1, learning_starts = 800,
                 tensorboard_log = '/home/flavio/Scrivania/Soft-Actor-Critic-implementation/logs') -> None:


        # TODO: add all the parameters to all variables
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tensorboard_log = tensorboard_log

        self.replay_buffer = ReplayBuffer(max_size=buffer_size, state_dim=environment.observation_space.shape[0], action_dim=environment.action_space.shape[0])
        self.q1_network = CriticNetwork(state_dim=environment.observation_space.shape, action_dim=environment.action_space.shape)
        self.q2_network = CriticNetwork(state_dim=environment.observation_space.shape, action_dim=environment.action_space.shape)
        self.value_network = ValueNetwork(state_dim=environment.observation_space.shape)
        self.policy = ActorNetwork(state_dim=environment.observation_space.shape[0], max_actions_values=environment.action_space.high, device=self.device)
        # self.alpha = AlphaNetwork()
        self.alpha = ent_coef
        self.env = environment

        self.criterion = nn.MSELoss()
        self.optimizer_q1 = optim.Adam(self.q1_network.parameters(), lr=lr)
        self.optimizer_q2 = optim.Adam(self.q2_network.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value_network.parameters(), lr=lr)
        self.optimizer_actor = optim.Adam(self.policy.parameters(), lr=lr)

        # Parameters of original SAC_value (from stable-baselines) I need
        # to make them compatible
        self.lr = lr
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.gradient_steps = gradient_steps
        self.learning_starts = learning_starts


    # Initialize all parameters of all networks
    # NOTE: q1_network_target has the same initial parameters of q1_network,
    #       q2_network_target has the same initial parameters of q2_network
    def initialize_networks_parameters(self):
        self.q1_network.apply(init_weights)
        self.q2_network.apply(init_weights)
        self.value_network.apply(init_weights)
        self.policy.apply(init_weights)

        # Target network must start equals to its corresponding V-network
        self.value_network_target = copy.deepcopy(self.value_network)
       


    # Update the parameters of the networks
    def update_parameters():
        pass

    def move_to_device(self):
        self.q1_network = self.q1_network.to(self.device)
        self.q2_network = self.q2_network.to(self.device)
        self.value_network = self.value_network.to(self.device)
        self.value_network_target = self.value_network_target.to(self.device)
        self.policy = self.policy.to(self.device)
        # self.alpha = self.alpha.to(self.device)

    # Return the dir name
    def get_logdir_name(self, tb_log_name):
        i = 0
        while os.path.exists(f"{os.path.join(self.tensorboard_log, tb_log_name)}_%s" % i):
            i += 1

        dir_name = tb_log_name + "_" + str(i)
        dir = os.path.join(self.tensorboard_log, dir_name)
        # print(dir)
        return dir
        
    # Function that makes everything work.
    # Params:   - timesteps
    def learn(self, total_timesteps=800, log_interval=1, tb_log_name="SAC_value"):
        # Create Logger for debugging, with incremental filename
        dir = self.get_logdir_name(tb_log_name=tb_log_name)
        self.writer = SummaryWriter(log_dir=dir)

        # 1-2:  Initialize parameters of the networks, and move to device
        self.initialize_networks_parameters()
        self.move_to_device()

        # DEBUG:
        # utils.check_model_device(self.policy)
        # utils.check_model_device(self.q1_network)
        # utils.check_model_device(self.q2_network)
        # utils.print_model_parameters(self.policy)

        # Total number of steps done in all the episodes
        num_total_steps = 0

        for i in range(total_timesteps):
            # Number of steps in the current episode
            num_steps_in_episode = 0
            # Reset the environment because it's a new episode
            state, _ = self.env.reset()
            done = False
            # It contains the sum of the rewards of the current episode
            total_reward = 0

            q1_loss_records = []
            q2_loss_records = []
            q_loss_records = []
            value_loss_records = []
            actor_loss_records = []

            # While the episode is not finished (terminated or truncated at 200 steps)
            while not done:
                print("step in episode:     ", num_steps_in_episode)
                print("total steps:     ", num_total_steps)
                
                # print("STATE:       ", state)
                # 4)    Save the state as a tensor (and send to device because the networks are there)
                state = torch.tensor(state, dtype=torch.float).to(self.device)
                # print("state:   ", state)

                # Render the state to visualize it
                # self.env.render()

                # 4)    Sample an action through Actor network
                # It returns a tensor. The env.step want a numpy array
                # utils.print_model_parameters(self.policy)
                # utils.check_gradients(self.policy)

                if num_total_steps < self.learning_starts:
                    action = np.random.uniform(low=self.env.action_space.low, high=self.env.action_space.high)
                else:
                    action,_ = self.policy.sample_action_logprob(state, reparam_trick = False)
                    action = action.to('cpu').detach().numpy()
                # NOTE: Tensor is on device (that can be also cuda), to convert it to numpy array
                #       it needs it's on cpu and requires_grad=False, so we need to detach it
                #       (https://stackoverflow.com/questions/49768306/pytorch-tensor-to-numpy-array)
                # print("ACTION:      ", action)
                # 5-6)  Make a step in the environment, and observe (next state, reward, done)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                # print("NEXT STATE:      ", next_state)      # CONTINUA DA QUI
                # print("TERMINATED:        ", terminated)
                # print("TRUNCATED:       ", truncated)
                total_reward += reward
                # print("next state:  ", next_state)
                # print("reward:  ", reward)
                # 7)    Store the transition in the replay buffer
                # Convert state from tensor to numpy arrayù
                state = state.to('cpu').detach().numpy()
                self.replay_buffer.store_transition(state, action, reward, next_state, done)
                
                # The current state becomes the next state
                state = next_state

                # 9)    If the episode (after action have been applied) is not finished
                if num_total_steps >= self.learning_starts:    # TODO: to remove
                    # 10)   For the number of update steps
                    for j in range(self.gradient_steps):
                        # 11)   Sample randomly a batch of transitions   
                        states_batch, actions_batch, next_states_batch, rewards_batch, dones_batch = self.replay_buffer.sample_from_buffer(self.batch_size)
                        # Convert them into tensors and send them to device
                        states = torch.tensor(states_batch, dtype=torch.float).to(self.device)
                        actions = torch.tensor(actions_batch, dtype=torch.float).to(self.device)
                        rewards = torch.tensor(rewards_batch, dtype=torch.float).unsqueeze(1).to(self.device)
                        dones = torch.tensor(dones_batch, dtype=torch.float).unsqueeze(1).to(self.device)
                        next_states = torch.tensor(next_states_batch, dtype=torch.float).to(self.device)
                        # print("next states:     ", next_states)
                        

                        actions_sampled, log_probs_sampled = self.policy.sample_action_logprob(states, reparam_trick=False)
                        q1_values = self.q1_network(states, actions_sampled)
                        q2_values = self.q2_network(states, actions_sampled)
                        q_min_values = torch.min(q1_values, q2_values)
                        y = q_min_values - log_probs_sampled
                        
                        self.optimizer_value.zero_grad()
                        value_output = self.value_network(states)
                        value_loss = 0.5 * self.criterion(value_output, y)
                        value_loss_records.append(value_loss.to('cpu').detach().numpy())
                        value_loss.backward(retain_graph=True)
                        self.optimizer_value.step()



                        # 14) Update of policy (Actor) through Gradient Ascent
                        self.optimizer_actor.zero_grad()

                        actions_policy, log_prob_policy = self.policy.sample_action_logprob(states, reparam_trick=True)
                        q1_values = self.q1_network(states, actions_policy)
                        q2_values = self.q2_network(states, actions_policy)
                        q_min_values = torch.min(q1_values, q2_values)
                        actor_loss = torch.mean(log_prob_policy - q_min_values)
                        actor_loss_records.append(actor_loss.to('cpu').detach().numpy())
                        actor_loss.backward(retain_graph=True)
                        self.optimizer_actor.step()



                        # 13) Update of Q-functions (Q1)
                        # NOTE: we maintain the computational graph in memory (retain_graph=True) because
                        # the output of Q1 is used in the update of the policy network 

                        q_hat_values = rewards + self.gamma * self.value_network_target(next_states)

                        self.optimizer_q1.zero_grad()
                        q1_output = self.q1_network(states, actions)
                        q1_loss = 0.5 * self.criterion(q1_output, q_hat_values)
                        q1_loss_records.append(q1_loss.to('cpu').detach().numpy())
                        # q1_loss.backward(retain_graph=True)

                        self.optimizer_q2.zero_grad()
                        q2_output = self.q2_network(states.detach(), actions.detach())
                        q2_loss = 0.5 * self.criterion(q2_output, q_hat_values)
                        q2_loss_records.append(q2_loss.to('cpu').detach().numpy())
                        # q2_loss.backward(retain_graph=True)

                        q_loss = q1_loss + q2_loss
                        q_loss_records.append(q_loss.to('cpu').detach().numpy())
                        q_loss.backward(retain_graph=True)
                        self.optimizer_q1.step()
                        self.optimizer_q2.step()


                        # 15) Update target networks (Q1 target)
                        with torch.no_grad():
                            for param, param_target in zip(self.value_network.parameters(), self.value_network_target.parameters()):
                                if param.requires_grad and param_target.requires_grad:
                                    param_target = self.tau * param + (1-self.tau) * param_target

                num_steps_in_episode += 1
                num_total_steps += 1

            if(len(q1_loss_records)>0 and len(q2_loss_records)>0 and len(actor_loss_records)>0 and len(value_loss_records)>0 and len(q_loss_records)>0):
                q1_loss_average = np.mean(q1_loss_records)
                q2_loss_average = np.mean(q2_loss_records)
                q_loss_average = np.mean(q_loss_records)
                actor_loss_average = np.mean(actor_loss_records)
                value_loss_average = np.mean(value_loss_records)

                self.writer.add_scalar("Q1-Loss per episode", q1_loss_average, i)        
                self.writer.add_scalar("Q2-Loss per episode", q2_loss_average, i)        
                self.writer.add_scalar("Q-Loss per episode", q_loss_average, i)        
                self.writer.add_scalar("Actor-Loss per episode", actor_loss_average, i)        
                self.writer.add_scalar("Value-Loss per episode", value_loss_average, i)
            self.writer.add_scalar("Total reward per episode", total_reward, i)        
            print("TOTAL:       ", total_reward)


if __name__=="__main__":
    # We are going to test the algorithm on Pendulum-v1,
    # because it is the most similar environment to RoboCup-v1.
    # The observation space is an array of elements, while 
    # the action space is of type Box (continuous space)
    env = gym.make('Pendulum-v1', render_mode='human')

    # NOTE:
    # Observations and actions are tuples. In our environments (Classic Control)
    # the tuples are composed by just one element ( ex: (3,) )
    print("obs shape:   ", env.observation_space.shape)
    print("action shape:   ", env.action_space.shape)

    model = SAC_value(buffer_size=1000000, environment=env)
    model.learn()

