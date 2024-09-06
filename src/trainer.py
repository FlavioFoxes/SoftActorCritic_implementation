import gymnasium as gym
from algorithm.SAC import SAC

def trainer():
    env = gym.make('Pendulum-v1')
    
    model = SAC(buffer_size=1000000, environment=env, ent_coef="auto")
    model.learn(num_episodes = 100)
            