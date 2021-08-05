import numpy as np
import matplotlib.pyplot as plt




def plot_single_state(x, y, phi, L):
    r_propeller_x = x + L*np.cos(phi)
    r_propeller_y = y + L*np.sin(phi)

    l_propeller_x = x - L*np.cos(phi)
    l_propeller_y = y - L*np.sin(phi)


    fig = plt.figure(1)
    plt.scatter(x, y, marker='s')
    plt.plot([x, r_propeller_x], [y, r_propeller_y])
    plt.plot([x, l_propeller_x], [y, l_propeller_y])
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.show()


def plot_multiple_states(x, y, phi, L, constr):
    fig, ax = plt.subplots()
    for (xk, yk, phik) in zip(x, y, phi):
        r_propeller_x = xk + L*np.cos(phik)
        r_propeller_y = yk + L*np.sin(phik)

        l_propeller_x = xk - L*np.cos(phik)
        l_propeller_y = yk - L*np.sin(phik)

        ax.scatter(xk, yk, marker='s', color = 'black')
        ax.plot([xk, r_propeller_x], [yk, r_propeller_y], 'b')
        ax.plot([xk, l_propeller_x], [yk, l_propeller_y], 'b')
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    plt.show()


L = 0.2
x = [1, 2]
y = [1, 2]
phi = [np.pi/4, np.pi/6]

plot_multiple_states(x, y, phi, L)
plt.show()