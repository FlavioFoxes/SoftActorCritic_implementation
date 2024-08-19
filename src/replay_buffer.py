import numpy as np
import random

# Replay buffer is a buffer containing all last N steps/episodes
# of the envirnoment. It is composed of different buffers:
#   - states:            state in which it starts
#   - actions:           action applied when it is in state
#   - rewards:           reward obtained passing from state to next_state
#   - next_states:       state where it arrives when applied action to state
#   - dones:             done flag when it is in state
#
# It has to contain two functions:
#   - store_transition [DONE]
#   - sample_from_buffer [DONE]
class ReplayBuffer:
    def __init__(self, max_size, shape, num_actions) -> None:
        self.state_buffer = np.zeros((max_size, shape))
        self.action_buffer = np.zeros((max_size, num_actions))  
        self.next_state_buffer = np.zeros((max_size, shape))
        self.reward_buffer = np.zeros(max_size)
        self.done_buffer = np.zeros(max_size, dtype=np.bool)

        self.current_buffer_index = 0
        self.buffer_size = max_size

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

    def sample_from_buffer(self, batch_size):
        # Maximum size which contains valid sample that could be extracted
        size = min(self.buffer_size, self.current_buffer_index)
        # Extract indices to be sampled from the buffers
        indices = random.sample(range(size), batch_size)

        # Extract samples from the buffers
        states_sampled = self.state_buffer[indices]
        actions_sampled = self.action_buffer[indices]
        next_states_sampled = self.next_state_state_buffer[indices]
        rewards_sampled = self.reward_buffer[indices]
        dones_sampled = self.done_buffer[indices]

        return states_sampled, actions_sampled, next_states_sampled, rewards_sampled, dones_sampled
