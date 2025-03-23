import gymnasium as gym
from src.algorithm.SAC import SAC

LOGS_PATH = '/home/flavio/Scrivania/RL_logs/'
# INFO: guardare l'esempio dell'environment del pendolo per vedere quando non ci sono 
# compatibilità di argomenti o cose simili

# Crea l'ambiente
env = gym.make('RoboCup-v1')

# policy_kwargs = dict(net_arch=[256, 256, 256, 256])  # default
# Crea il modello PPO
model = SAC(environment=env, ent_coef="auto", lr=3e-3, batch_size= 3, learning_starts=4, tensorboard_log=LOGS_PATH)

# Addestra il modello. Se nel mentre viene interrotto, chiudi tutte le socket
# TODO: per ora, continua ad uccidere il processo a mano, usando i comandi
# lsof -i :12345
# kill <pid>
try:
    model.learn(num_episodes=1000, tb_log_name="Gaussian_logs")
    model.save("sac_robocup")
    # Chiudi tutte le socket
    env.close()
except KeyboardInterrupt:
    # Chiudi tutte le socket
    env.close()


# Esempio di utilizzo dell'ambiente
# env = gym.make('RoboCup-v1')
# obs = env.reset()
# done = False

# action = env.action_space.sample()  # Esegui un'azione casuale
# obs, reward, done, info = env.step(action)
# env.render()