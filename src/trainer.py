import gymnasium as gym
from src.algorithm.SAC import SAC

def trainer():
    env = gym.make('Pendulum-v1', render_mode = "human")
    
    model = SAC(buffer_size=1000, environment=env, ent_coef=0.2, learning_starts=100, batch_size=50)
    model.learn(num_episodes = 100)
            