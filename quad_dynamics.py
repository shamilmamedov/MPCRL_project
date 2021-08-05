import casadi as c
# from helpers import integrate_RK4

# Parameters
g = 9.81
dt = 0.01
m = 1
L = 0.2
Izz = 0.1


# Tuning stage cost
Q = c.diag([1, 1, 1, 1, 1, 1])
R = c.diag([0.1, 0.1])

# State
x = c.MX.sym('x', 6)
px = x[0]
py = x[1]
phi = x[4]
vx = x[2]
vy = x[3]
phidot = x[5]

# Control
u = c.MX.sym('u', 2)
F1 = u[0]
F2 = u[1]

# Time
T = c.MX.sym('T')


# Dynamics
ax = -1/m * (F1 + F2) * c.sin(phi)
ay = 1/m * (F1 + F2) * c.cos(phi) - g
phiddot = 1/Izz * (F1 - F2) * L
ode = c.vertcat(vx, vy, phidot, ax, ay, phiddot)

# Cost
L = x.T @ Q @ x + u.T @ R @ u


# Extend for FreeT problem
x = c.vertcat(x, T)
ode = c.vertcat(T*ode, 0)


def get_dynamics():
    return ode, x, u, L
