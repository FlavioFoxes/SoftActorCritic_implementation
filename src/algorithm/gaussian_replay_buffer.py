import numpy as np
import random

'''
 Replay buffer is a buffer containing all last N steps/episodes
 of the envirnoment. It is composed of different buffers:
   - states:            state in which it starts
   - actions:           action applied when it is in state
   - rewards:           reward obtained passing from state to next_state
   - next_states:       state where it arrives when applied action to state
   - dones:             done flag when it is in state
'''
class GaussianReplayBuffer:
    '''
        Parameters:
        -   max_size:               maximum size of the buffer
        -   state_dim:              dimension of the state
        -   action_dim:             dimension of the action
    '''
    
    def __init__(self, max_size, state_dim, action_dim) -> None:
        self.state_buffer = np.zeros((max_size, state_dim))
        self.action_buffer = np.zeros((max_size, action_dim))  
        self.next_state_buffer = np.zeros((max_size, state_dim))
        self.reward_buffer = np.zeros(max_size)
        self.done_buffer = np.zeros(max_size, dtype=bool)

        # Represents the index of last element of the buffer
        self.current_buffer_index = 0
        # Represents the total size of the buffer
        self.buffer_size = max_size



    def find_index(self, reward):
        # Middle transition of the buffer
        left = 0
        right = self.current_buffer_index
        mid = 0
            
        while left < right:
            mid = (left + right) // 2
            if reward > self.reward_buffer[mid]:
                right = mid
            else:
                left = mid + 1
        # mid = right if reward > self.reward_buffer[right] else left
        return left



    '''
    Store a transition in the buffer
        Parameters:
        - state:                    starting state of the transition
        - action:                   action applied in the starting state
        - reward:                   reward obtained passing from state to next_state
        - next_state:               ending state of the transition
        - done:                     (bool) if episode is terminated
    '''
    def store_transition(self, state, action, reward, next_state, done):
        # This is done because, after the buffer is full,
        # it restarts writing from the beginning

        i = self.find_index(reward)        
        # buffer[index+1:] = buffer[index:-1]

        # Insert in the i-th position of the buffers the element
        self.state_buffer[i+1:] = self.state_buffer[i:-1]
        self.state_buffer[i] = state

        self.action_buffer[i+1:] = self.action_buffer[i:-1]
        self.action_buffer[i] = action

        self.reward_buffer = np.insert(self.reward_buffer, i, reward)
        
        self.next_state_buffer[i+1:] = self.next_state_buffer[i:-1]
        self.next_state_buffer[i] = next_state
        
        self.done_buffer = np.insert(self.done_buffer, i, done)


        # self.state_buffer[i+1:] = self.state_buffer[i:-1]
        # self.state_buffer[i] = state

        # self.action_buffer = np.insert(self.action_buffer, i, action)
        # self.next_state_buffer = np.insert(self.next_state_buffer, i, next_state)

        # # Delete last element of the buffers because they increased by one after the insert
        # self.state_buffer = np.delete(self.state_buffer, -1)
        # self.action_buffer = np.delete(self.action_buffer, -1)
        self.reward_buffer = np.delete(self.reward_buffer, -1)
        # self.next_state_buffer = np.delete(self.next_state_buffer, -1)
        self.done_buffer = np.delete(self.done_buffer, -1)

        # Increase index of the last element of the buffers
        if self.current_buffer_index < self.buffer_size-1:
            self.current_buffer_index += 1 

        # print("Reward size:     ", self.reward_buffer.shape)
        # print("REWARD BUFFER:  ", self.reward_buffer[:self.current_buffer_index])

    '''
    Sample from buffer a batch of transitions
        Parameters:
        - batch_size:               batch size of transitions to sample
        Returns:
        - states_sampled:           starting states of the sampled transitions
        - actions_sampled:          actions applied in the starting sampled states  
        - next_states_sampled:      ending states of the sampled transitions
        - rewards_sampled:          rewards obtained passing from sampled states to sampled next_states
        - dones_sampled:            (bool) if corresponding episodes are the final ones
    '''
    def sample_from_buffer(self, batch_size):

        # Mean and standard deviation of Gaussian distribution
        mu = 0
        sigma = self.current_buffer_index // 3

        # Maximum size which contains valid sample that could be extracted
        size = min(self.buffer_size, self.current_buffer_index)

        indices = np.arange(size)
        gaussian_probs = np.exp(-0.5 * ((indices - mu) / sigma) ** 2)

        # Normalizza le probabilità
        gaussian_probs /= gaussian_probs.sum()

        # From the array of indices, sample a batch following a gaussian PDF
        samples = np.random.choice(indices, size=batch_size, p=gaussian_probs, replace=False)

        # Extract samples from the buffers
        states_sampled = self.state_buffer[samples]
        actions_sampled = self.action_buffer[samples]
        next_states_sampled = self.next_state_buffer[samples]
        rewards_sampled = self.reward_buffer[samples]
        dones_sampled = self.done_buffer[samples]

        return states_sampled, actions_sampled, next_states_sampled, rewards_sampled, dones_sampled