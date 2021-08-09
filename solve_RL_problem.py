import os
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from Quad2d_env import Drone

from stable_baselines3.common import results_plotter
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

train_new_model = False
time_steps = 5e+5

# define enivronment
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

env = Drone()
env = Monitor(env, log_dir)
env = DummyVecEnv([lambda: env])
env.reset()

# Soft-Actor Critic
if train_new_model:
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

    policy_kwargs = dict(net_arch = dict(pi = [256, 256, 256], qf = [256, 256, 256]))

    sac_model = SAC("MlpPolicy", env, verbose=1, learning_rate=1e-3, 
                    tensorboard_log="logs/sac_planardrone_tensorboard/",
                    action_noise=action_noise, policy_kwargs=policy_kwargs)

    sac_model.learn(total_timesteps=time_steps, log_interval=10, reset_num_timesteps=True)     
else:
    sac_model = SAC.load("sac_drone", env=env, tensorboard_log="logs/sac_planardrone_tensorboard/")
    sac_model.load_replay_buffer("sac_drone_replay_buffer")
    # Pass reset_num_timesteps=False to continue the training curve in tensorboard
    sac_model.learn(total_timesteps=time_steps, log_interval=10, reset_num_timesteps=False)     



# plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "SAC PlanarDrone")
# plt.show()

# x, y = ts2xy(load_results(log_dir), 'timesteps')
# fig = plt.figure()
# plt.plot(x, y)
# plt.show()

sac_model.save("sac_drone")
sac_model.save_replay_buffer("sac_drone_replay_buffer")

# del model # remove to demonstrate saving and loading

# # TD3
# # n_actions = env.action_space.shape[-1]
# # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# # td3_model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
# # td3_model.learn(total_timesteps=100000, log_interval=100)
# # td3_model.save("td3_drone")
# # # env = td3_model.get_env()