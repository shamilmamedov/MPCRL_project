import os
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from quadrotor_env import PlanarQuadrotorEnv

from stable_baselines3.common import results_plotter
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

def train_new_SAC_model(env, time_steps=1e+5):
    # define action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # define policy parameters: network architecture
    policy_kwargs = dict(net_arch = dict(pi = [128, 128, 128], qf = [512, 256, 128]))
    lr = 1e-3

    # create SAC model
    sac_model = SAC("MlpPolicy", env, verbose=1, learning_rate=lr, gamma=1,
                    tensorboard_log="logs/sac_planardrone_tensorboard/",
                    action_noise=action_noise, policy_kwargs=policy_kwargs)

    # train model
    sac_model.learn(total_timesteps=time_steps, log_interval=25, reset_num_timesteps=True) 

    # save model and replay buffer
    sac_model.save("RL_models/sac_drone")
    sac_model.save_replay_buffer("RL_models/sac_drone_replay_buffer")


def continue_traininig_SAC_model(env, time_steps=1e+5):
    # load saved model
    sac_model = SAC.load("RL_models/sac_drone", env=env, tensorboard_log="logs/sac_planardrone_tensorboard/")
    sac_model.load_replay_buffer("RL_models/sac_drone_replay_buffer")

    # Pass reset_num_timesteps=False to continue the training curve in tensorboard
    sac_model.learn(total_timesteps=time_steps, log_interval=25, reset_num_timesteps=False) 

    # save model and replay buffer
    sac_model.save("RL_models/sac_drone")
    sac_model.save_replay_buffer("RL_models/sac_drone_replay_buffer")


def train_new_TD3_model(env, time_steps=1e+5):
    # define action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    # define policy parameters: network architecture
    policy_kwargs = dict(net_arch = dict(pi = [128, 128], qf = [400, 300]))
    lr = 1e-3
    # policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[400, 300]))
    
    # Create TD3 model
    td3_model = TD3("MlpPolicy", env, verbose=1, learning_rate=lr, gamma=1, 
                    tensorboard_log="logs/td3_planardrone_tensorboard/",
                    action_noise=action_noise, policy_kwargs=policy_kwargs)

    # train model
    td3_model.learn(total_timesteps=time_steps, log_interval=25, reset_num_timesteps=True)

    # save model and replay buffer
    td3_model.save("RL_models/td3_drone")
    td3_model.save_replay_buffer("RL_models/td3_drone_replay_buffer")


def continue_traininig_TD3_model(env, time_steps=1e+5):
    # load saved model
    td3_model = TD3.load("RL_models/td3_drone", env=env, tensorboard_log="logs/td3_planardrone_tensorboard/")
    td3_model.load_replay_buffer("RL_models/td3_drone_replay_buffer")

    # Pass reset_num_timesteps=False to continue the training curve in tensorboard
    td3_model.learn(total_timesteps=time_steps, log_interval=25, reset_num_timesteps=False) 

    # save model and replay buffer
    td3_model.save("RL_models/td3_drone")
    td3_model.save_replay_buffer("RL_models/td3_drone_replay_buffer")


time_steps = 5e+5

# define enivronment
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

env = PlanarQuadrotorEnv()
env = Monitor(env, log_dir)
env = DummyVecEnv([lambda: env])
env.reset()

# train_new_SAC_model(env, time_steps=1e+4)
# continue_traininig_SAC_model(env, time_steps)
# train_new_TD3_model(env)
continue_traininig_TD3_model(env, time_steps)


# plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "SAC PlanarDrone")
# plt.show()

# x, y = ts2xy(load_results(log_dir), 'timesteps')
# fig = plt.figure()
# plt.plot(x, y)
# plt.show()
