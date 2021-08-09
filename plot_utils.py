import matplotlib.pyplot as plt
from casadi import cos, sin, pi, linspace

def visualize_trajectory(x, y, phi, L, obstacles=None):
    fig, ax = plt.subplots()
    for (xk, yk, phik) in zip(x, y, phi):
        r_propeller_x = xk + L*cos(phik)
        r_propeller_y = yk + L*sin(phik)

        l_propeller_x = xk - L*cos(phik)
        l_propeller_y = yk - L*sin(phik)

        ax.scatter(xk, yk, marker='s', color = 'black')
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

def plot_controls(tgrid, u):
    fig, ax = plt.subplots()
    ax.step(tgrid[0:-1], u[0,:], where='post', label=r'$u_1$')
    ax.step(tgrid[0:-1], u[1,:], where='post', label=r'$u_2$')
    ax.set_xlabel(r'$t$ (sec)')
    ax.set_ylabel(r'$u$ (N)')
    ax.legend()
    ax.grid()
    plt.show()

def plot_pose(tgrid, pose):
    fig, ax = plt.subplots()
    ax.plot(tgrid, pose[0,:], label=r'$x$')
    ax.plot(tgrid, pose[1,:], label=r'$y$')
    ax.plot(tgrid, pose[2,:], label=r'$\phi$')
    ax.set_xlabel(r'$t$ (sec)')
    ax.set_ylabel(r'$u$ (N)')
    ax.legend()
    ax.grid()
    plt.show()