import gymnasium as gym
from src.algorithm.SAC import SAC

def trainer():
    env = gym.make('Pendulum-v1')
    
    model = SAC(buffer_size=1000000, environment=env, ent_coef=0.2)
    model.learn(num_episodes = 100)
            