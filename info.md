# SAC
## Idea
The idea behind **Soft Actor-Critic** is to optimize a stochastic policy, which takes into account both maximize the expected reward and  entropy, following Actor-Critic approach. This is useful for the compromise of exploration-exploitaiton.\
The **Actor** is the network that represents the policy, so it decides the action to be done basing on the state of the environment.\
The **Critic** is the network that evaluates the goodness of the action chosen by the Actor. The Critic is the Q-network and here we use two different Critics. Then we choose the minimum value between them for the updates.\
The **Value** network I don't know if it is necessary.\
The **Target** network is used for the updates, because of the update of Q-network depends on the V-network, and update of V-network depends on Q-network. So V-network depends indirectly on itself. So the target V-function is used, which which is very close to the main V-function, but with a time-delay.


## Implementation
### Elements
- **Replay Buffer** `[x]`
- **Actor network** `[?]`
- **Q1 network** `[?]`
- **Q2 network** `[?]`
- **Value network** `[]`\
(I think it's not necessary, depends on the implementation)
- **Target network** `[?]`\
(w.r.t. Value/Q networks)
- **Alpha Network** `[]`

### Notes
- **Q1** and **Q2** are the Critic networks. They are implemented, but must be checked
- At the moment, the idea is to use the target networks w.r.t. the **Q** networks. So we need two **Target networks** 


### Specifics
- **Replay Buffer**: it has to contain all the transitions **(state, action, reward, next state,done)**; we need two functions, `store_transition` to memorize a transition in the buffer, `sample_from_buffer` to extract a batch of the memorized samples

- **Actor network**: represents the policy. It takes in input the state and it returns the mean and variance of a normal distribution, which represent the stochastic action.\
It contains also a function to sample the action using the mean and the variance, and returns the action and the log prob

- **Critic network**: it is a Q-network. It takes in input the state and the action, and returns a value which represents the goodness of the chosen action in the given state. It could be used also for the target network if Value network is not implemented (see *Soft Actor-Critic Algorithms and Applications*)

- **Value network**: it takes in input the state and returns a value which represents the goodness of that state. It could be used also for the target network (see *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*)

- **Target network**: see ***Value network***

- **Alpha network**: network for the adjustment of the entropy temperature