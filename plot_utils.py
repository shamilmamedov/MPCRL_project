import matplotlib.pyplot as plt
from casadi import cos, sin, pi, sqrt, linspace
import matplotlib
from pandas import read_csv
import numpy as np


def latexify(fig_width=None, fig_height=None):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    if fig_width is None:
        fig_width = 5  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches


    params = {'backend': 'ps',
              'text.latex.preamble': r"\usepackage{gensymb} \usepackage{amsmath}",
              'axes.labelsize': 10, # fontsize for x and y labels (was 10)
              'axes.titlesize': 10,
              'legend.fontsize': 10, # was 10
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'text.usetex': True,
              'figure.figsize': [fig_width, fig_height],
              'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)


def visualize_trajectory(x, y, phi, L, obstacles=None, save_fig=False, fig_name=None):
    latexify(fig_width=3, fig_height=3)
    marker_size = 5
    fig, ax = plt.subplots()
    ax.scatter(9, 9, marker='x')
    for (xk, yk, phik) in zip(x, y, phi):
        r_propeller_x = xk + L*cos(phik)
        r_propeller_y = yk + L*sin(phik)

        l_propeller_x = xk - L*cos(phik)
        l_propeller_y = yk - L*sin(phik)

        ax.scatter(xk, yk, marker_size, marker='s', color = 'black')
        ax.plot([xk, r_propeller_x], [yk, r_propeller_y], 'b')
        ax.plot([xk, l_propeller_x], [yk, l_propeller_y], 'b')

    if obstacles != None:
        for o in obstacles:
            px_obst_k = o[0][0]
            py_obst_k = o[0][1]
            r_obst_k = o[1]
            theta = linspace(0, 2*pi, 100)
            xobst1 = px_obst_k + r_obst_k*cos(theta)
            xobst2 = py_obst_k + r_obst_k*sin(theta)
            plt.plot(xobst1, xobst2, 'r')

    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect(1)
    plt.show()

    if save_fig:
        fig.savefig(fig_name + '.svg', format='svg', dpi=1200)

def plot_controls(tgrid, u, save_fig=False, fig_name=None):
    latexify(fig_width=2, fig_height=1.5)
    fig, ax = plt.subplots()
    ax.step(tgrid[0:-1], u[0,:], where='post', label=r'$F_1$')
    ax.step(tgrid[0:-1], u[1,:], where='post', label=r'$F_2$')
    ax.set_xlabel(r'$k$ (iteration)')
    ax.set_ylabel(r'$F$ (N)')
    ax.legend()
    ax.grid()
    plt.show()

    if save_fig:
        fig.savefig(fig_name + '.svg', format='svg', dpi=1200)

def plot_pose(tgrid, pose, save_fig=False):
    latexify()
    fig, ax = plt.subplots()
    ax.plot(tgrid, pose[0,:], label=r'$x$')
    ax.plot(tgrid, pose[1,:], label=r'$y$')
    ax.plot(tgrid, pose[2,:], label=r'$\phi$')
    ax.set_xlabel(r'$t$ (sec)')
    ax.set_ylabel(r'$u$ (N)')
    ax.legend()
    ax.grid()
    plt.show()

    if save_fig:
        fig.savefig('drone_pose.svg', format='svg', dpi=1200)


def plot_RL_statistics(save_fig=False):
    latexify(fig_width=11, fig_height=4)

    # load trainig statistics
    sac_reward = read_csv('report/SAC7_reward.csv')
    td3_reward = read_csv('report/TD36_reward.csv')

    sac_actor_loss = read_csv('report/SAC7_actor_loss.csv')
    td3_actor_loss = read_csv('report/TD36_actor_loss.csv')

    sac_critic_loss = read_csv('report/SAC7_critic_loss.csv')
    td3_critic_loss = read_csv('report/TD36_critic_loss.csv')


    fig, ax = plt.subplots(1,3)
    ax[0].plot(1e-5*sac_reward.values[:,1], 1e-3*sac_reward.values[:,2], label='SAC')
    ax[0].plot(1e-5*td3_reward.values[:,1], 1e-3*td3_reward.values[:,2], label='TD3')
    ax[0].set_xlabel(r'time steps $\times\ 10^5$')
    ax[0].set_ylabel(r'episode return $\times\ 10^3$')
    ax[0].legend()

    ax[1].plot(1e-5*sac_actor_loss.values[:,1], 1e-3*sac_actor_loss.values[:,2], label='SAC')
    ax[1].plot(1e-5*td3_actor_loss.values[:,1], 1e-3*td3_actor_loss.values[:,2], label='TD3')
    ax[1].set_xlabel(r'time steps $\times\ 10^5$')
    ax[1].set_ylabel(r'actor loss $\times\ 10^3$')
    ax[1].legend()

    ax[2].plot(1e-5*sac_critic_loss.values[:,1], 1e-4*sac_critic_loss.values[:,2], label='SAC')
    ax[2].plot(1e-5*td3_critic_loss.values[:,1], 1e-4*td3_critic_loss.values[:,2], label='TD3')
    ax[2].set_xlabel(r'time steps $\times\ 10^5$')
    ax[2].set_ylabel(r'critic loss $\times\ 10^4$')
    ax[2].set_ylim(-2, 50)
    ax[2].legend()

    plt.show()

    if save_fig:
        fig.savefig('report/RL_statistics.svg', format='svg', dpi=1200)


def plot_controls_RL(t_grid_td3, u_traj_td31, t_grid_sac, u_traj_sac1, save_fig=False):
    latexify(fig_width=2, fig_height=1.5)
    fig, ax = plt.subplots()
    ax.step(t_grid_td3[0:-1], u_traj_td31[:,0], where='post', label=r'$F_1$ TD3')
    ax.step(t_grid_td3[0:-1], u_traj_td31[:,1], where='post', label=r'$F_2$ TD3')
    ax.step(t_grid_sac[0:-1], u_traj_sac1[:,0], '--', where='post', label=r'$F_1$ SAC')
    ax.step(t_grid_sac[0:-1], u_traj_sac1[:,1], '--', where='post', label=r'$F_2$ SAC')
    ax.set_xlabel(r'$k$ (iteration)')
    ax.set_ylabel(r'$F$ (N)')
    ax.legend()
    ax.grid()
    plt.show()
    if save_fig:
            fig.savefig('report/controls_RL.svg', format='svg', dpi=1200)

def visualize_trajectory_RL(x_traj_td31, x_traj_td32, x_traj_sac1, x_traj_sac2, L, save_fig=False):
    latexify(fig_width=3, fig_height=3)
    marker_size = 5
    fig, ax = plt.subplots()
    ax.scatter(9, 9, marker='x')
    for (xk, yk, phik) in zip(x_traj_td31[::15,0], x_traj_td31[::15,1], x_traj_td31[::15,2]):
        r_propeller_x = xk + L*np.cos(phik)
        r_propeller_y = yk + L*np.sin(phik)

        l_propeller_x = xk - L*np.cos(phik)
        l_propeller_y = yk - L*np.sin(phik)

        ax.scatter(xk, yk, marker_size, marker='s', color = 'black')
        ax.plot([xk, r_propeller_x], [yk, r_propeller_y], 'b')
        ax.plot([xk, l_propeller_x], [yk, l_propeller_y], 'b')

    for (xk, yk, phik) in zip(x_traj_td32[::15,0], x_traj_td32[::15,1], x_traj_td32[::15,2]):
        r_propeller_x = xk + L*np.cos(phik)
        r_propeller_y = yk + L*np.sin(phik)

        l_propeller_x = xk - L*np.cos(phik)
        l_propeller_y = yk - L*np.sin(phik)

        ax.scatter(xk, yk, marker_size, marker='s', color = 'black')
        ax.plot([xk, r_propeller_x], [yk, r_propeller_y], 'b')
        ax.plot([xk, l_propeller_x], [yk, l_propeller_y], 'b')

    for (xk, yk, phik) in zip(x_traj_sac1[::15,0], x_traj_sac1[::15,1], x_traj_sac1[::15,2]):
        r_propeller_x = xk + L*np.cos(phik)
        r_propeller_y = yk + L*np.sin(phik)

        l_propeller_x = xk - L*np.cos(phik)
        l_propeller_y = yk - L*np.sin(phik)

        ax.scatter(xk, yk, marker_size, marker='s', color = 'black')
        ax.plot([xk, r_propeller_x], [yk, r_propeller_y], 'r')
        ax.plot([xk, l_propeller_x], [yk, l_propeller_y], 'r')

    for (xk, yk, phik) in zip(x_traj_sac2[::15,0], x_traj_sac2[::15,1], x_traj_sac2[::15,2]):
        r_propeller_x = xk + L*np.cos(phik)
        r_propeller_y = yk + L*np.sin(phik)

        l_propeller_x = xk - L*np.cos(phik)
        l_propeller_y = yk - L*np.sin(phik)

        ax.scatter(xk, yk, marker_size, marker='s', color = 'black')
        ax.plot([xk, r_propeller_x], [yk, r_propeller_y], 'r')
        ax.plot([xk, l_propeller_x], [yk, l_propeller_y], 'r')

    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect(1)
    plt.show()

    if save_fig:
            fig.savefig('report/trajectory_RL.svg', format='svg', dpi=1200)