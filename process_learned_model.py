import numpy as np
import time

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from quadrotor_env import PlanarQuadrotorEnv

env = PlanarQuadrotorEnv()
env = Monitor(env)
env = DummyVecEnv([lambda: env])

# model = SAC.load("RL_models/sac_obstacles_drone", env=env)
model = TD3.load("RL_models/td3_drone", env=env)

# initial_state = np.array([1,1,0,0,0,0])
# env.state = initial_state
# obs = initial_state

obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    # print(action, type(action))
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(.01)
    if done:
        # env.state = initial_state
        # obs = initial_state
        obs = env.reset()

env.close()