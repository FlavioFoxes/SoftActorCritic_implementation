import numpy as np


# Replay buffer is a buffer containing all last N steps/episodes
# of the envirnoment. It is composed of different buffers:
#   - states:            state in which it starts
#   - actions:           action applied when it is in state
#   - rewards:           reward obtained passing from state to next_state
#   - next_states:       state where it arrives when applied action to state
#   - dones:             done flag when it is in state
class ReplayBuffer:
    def __init__(self, max_size, shape, num_actions) -> None:
        self.state_buffer = np.zeros((max_size, shape))
        self.action_buffer = np.zeros((max_size, num_actions))  # maybe it is not shape but num_actions
        self.next_state_buffer = np.zeros((max_size, shape))
        self.reward_buffer = np.zeros(max_size)
        self.done_buffer = np.zeros(max_size, dtype=np.bool)

        self.buffer_index = 0

    