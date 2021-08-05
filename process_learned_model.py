import numpy as np
from Quad2d_env import Drone
from stable_baselines3 import SAC, TD3

env = Drone()
model = SAC.load("sac_drone", env=env)
# model = TD3.load("td3_drone", env=env)

initial_state = np.array([1,1,0,0,0,0])
env.state = initial_state
obs = initial_state
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        env.state = initial_state
        obs = initial_state

env.close()