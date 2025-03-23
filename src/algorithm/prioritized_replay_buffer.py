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
class PrioritizedReplayBuffer:
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
        i = self.current_buffer_index % self.buffer_size    

        # Store in the i-th index of the buffers the element
        self.state_buffer[i] = state
        self.action_buffer[i] = action
        self.next_state_buffer[i] = next_state
        self.reward_buffer[i] = reward
        self.done_buffer[i] = done

        self.current_buffer_index += 1


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
        # Maximum size which contains valid sample that could be extracted
        size = min(self.buffer_size, self.current_buffer_index)
        # Extract indices to be sampled from the buffers
        indices = random.sample(range(size), batch_size)
        # Extract samples from the buffers
        states_sampled = self.state_buffer[indices]
        actions_sampled = self.action_buffer[indices]
        next_states_sampled = self.next_state_buffer[indices]
        rewards_sampled = self.reward_buffer[indices]
        dones_sampled = self.done_buffer[indices]

        return states_sampled, actions_sampled, next_states_sampled, rewards_sampled, dones_sampled
