import matplotlib.pyplot as plt
from casadi import cos, sin, pi, sqrt, linspace
import matplotlib

def visualize_trajectory(x, y, phi, L, obstacles=None, save_fig=False, fig_name=None):
    latexify(fig_width=3, fig_height=3)
    marker_size = 5
    fig, ax = plt.subplots()
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
    # plt.show()

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