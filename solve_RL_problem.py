import gym
import numpy as np

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from Quad2d_env import Drone

env = Drone()

# Soft-Actor Critic
sac_model = SAC("MlpPolicy", env, verbose=1, learning_rate=0.001)
sac_model.learn(total_timesteps=100000, log_interval=10)

sac_model.save("sac_drone")
# del model # remove to demonstrate saving and loading

# TD3
# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# td3_model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
# td3_model.learn(total_timesteps=100000, log_interval=100)
# td3_model.save("td3_drone")
# # env = td3_model.get_env()
