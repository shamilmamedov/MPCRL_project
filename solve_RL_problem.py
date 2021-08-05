import os
import gym
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


class PlottingCallback(BaseCallback):
    """
    Callback for plotting the perofrmance in realtime
    """
    def __init__(self, log_dir, verbose=1):
        super(PlottingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self._plot = None
        

    def _on_step(self) -> bool:
        # get the monitor's data
        x, y = ts2xy(load_results(self.log_dir), 'timesteps')
        if self._plot is None:
            plt.ion()
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111)
            line, = ax.plot(x,y)
            self._plot = (line, ax, fig)
            plt.show()
        else:
            self._plot[0].set_data(x,y)
            self._plot[-2].relim()
            # self._plot[-2].set_xlim([-self.locals["total_timesteps"],
            #                          self.locals["total_timesteps"]])
            self._plot[-2].autoscale_view(True, True, True)
            self._plot[-1].canvas.draw()
# define enivronment
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

env = Drone()
env = Monitor(env, log_dir)
env = DummyVecEnv([lambda: env])


# Soft-Actor Critic
sac_model = SAC("MlpPolicy", env, verbose=1, learning_rate=1e-3)

# plotting_callback = PlottingCallback(log_dir)
time_steps = 5e+3
sac_model.learn(total_timesteps=time_steps, log_interval=10)     

plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "SAC PlanarDrone")
plt.show()

x, y = ts2xy(load_results(log_dir), 'timesteps')
fig = plt.figure()
plt.plot(x, y)
plt.show()

sac_model.save("sac_drone")
# del model # remove to demonstrate saving and loading

# # TD3
# # n_actions = env.action_space.shape[-1]
# # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# # td3_model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
# # td3_model.learn(total_timesteps=100000, log_interval=100)
# # td3_model.save("td3_drone")
# # # env = td3_model.get_env()