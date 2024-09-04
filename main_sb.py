import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

# Crea l'ambiente Pendulum-v1
env = gym.make('Pendulum-v1')

# Crea il modello SAC
model = SAC('MlpPolicy', env, verbose=1)

# Addestra il modello per 100.000 step
model.learn(total_timesteps=8000)

# Salva il modello
model.save("sac_pendulum_1")

# ----------------------------------------


# env = gym.make('Pendulum-v1', render_mode = 'human')

# # Carica il modello
# model = SAC.load("sac_pendulum_1", env=env)



# # Valuta il modello
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
# print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# # Enjoy trained agent
# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = vec_env.step(action)
#     vec_env.render("human")