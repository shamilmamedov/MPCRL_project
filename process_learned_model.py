import numpy as np
import matplotlib.pyplot as plt
 
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from quadrotor_env import PlanarQuadrotorEnv
from plot_utils import plot_RL_statistics, plot_controls_RL, visualize_trajectory_RL




# plot_RL_statistics(True)


initial_state1 = np.array([[1,1,0,0,0,0]])
initial_state2 = np.array([[9,1.5,0,0,0,0]])
save_fig = True


env = PlanarQuadrotorEnv(phi_range=np.pi)
env.reset()
model_td3 = TD3.load("RL_models/td3_drone", env=env)

env.state = initial_state1
obs = initial_state1

# run eposide with TD3 agent
x_traj_td31 = obs
u_traj_td31 = np.empty(shape=[0,2])
for _ in range(500):
    action, _states = model_td3.predict(obs, deterministic=True)
    u_traj_td31 = np.vstack((u_traj_td31, action))
    obs, reward, done, info = env.step(action)
    x_traj_td31 = np.vstack((x_traj_td31, obs))
    if done: break

env.reset()
env.state = initial_state2
obs = initial_state2

x_traj_td32 = obs
u_traj_td32 = np.empty(shape=[0,2])
for _ in range(500):
    action, _states = model_td3.predict(obs, deterministic=True)
    u_traj_td32 = np.vstack((u_traj_td32, action))
    obs, reward, done, info = env.step(action)
    x_traj_td32 = np.vstack((x_traj_td32, obs))
    if done: break
env.close()



# Recreate environment and run SAC agent
env = PlanarQuadrotorEnv(phi_range=np.pi)
env.reset()
model_sac = SAC.load("RL_models/sac_drone", env=env)

env.state = initial_state1
obs = initial_state1

x_traj_sac1 = obs
u_traj_sac1 = np.empty(shape=[0,2])
for _ in range(500):
    action, _states = model_sac.predict(obs, deterministic=True)
    u_traj_sac1 = np.vstack((u_traj_sac1, action))
    obs, reward, done, info = env.step(action)
    x_traj_sac1 = np.vstack((x_traj_sac1, obs))
    if done: break

env.reset()
env.state = initial_state2
obs = initial_state2

x_traj_sac2 = obs
u_traj_sac2 = np.empty(shape=[0,2])
for _ in range(500):
    action, _states = model_sac.predict(obs, deterministic=True)
    u_traj_sac2 = np.vstack((u_traj_sac2, action))
    obs, reward, done, info = env.step(action)
    x_traj_sac2 = np.vstack((x_traj_sac2, obs))
    if done: break
env.close()

# process trajectories
episode_len_td3 = x_traj_td31.shape[0]
t_grid_td3 = np.arange(0, episode_len_td3, 1)

episode_len_sac = x_traj_sac1.shape[0]
t_grid_sac = np.arange(0, episode_len_sac, 1)


plot_controls_RL(t_grid_td3, u_traj_td31, t_grid_sac, u_traj_sac1, save_fig=True)
visualize_trajectory_RL(x_traj_td31, x_traj_td32, x_traj_sac1, x_traj_sac2, env.arm_length, save_fig=True)

