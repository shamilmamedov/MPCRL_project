import casadi as csd
# from helpers import integrate_RK4
import matplotlib.pyplot as plt
import numpy as np

class PlanarDroneDynamics:
    def __init__(self):
        self.g = 9.81 
        self.m = 1
        self.Izz = 0.1
        self.L = 0.2
        
        self.x = csd.MX.sym('x', 6)
        self.u = csd.MX.sym('u', 2)

        px = self.x[0]
        py = self.x[1]
        phi = self.x[4]
        vx = self.x[2]
        vy = self.x[3]
        phidot = self.x[5]

        F1 = self.u[0]
        F2 = self.u[1]

        ax = -1/self.m * (F1 + F2) * csd.sin(phi)
        ay = 1/self.m * (F1 + F2) * csd.cos(phi) - self.g
        phiddot = 1/self.Izz * (F1 - F2) * self.L
        self.ode = csd.vertcat(vx, vy, phidot, ax, ay, phiddot)


class OptimalControlProblem:
    def __init__(self, J, w, g, w0, lbw, ubw, lbg, ubg):
        self.J = J
        self.w = w
        self.g = g
        self.w0 = w0
        self.lbw = lbw
        self.ubw = ubw
        self.lbg = lbg
        self.ubg = ubg


class PlanarDroneOCP:
    def __init__(self, x0 = [1, 1, 0, 0, 0, 0], xT = [5, 2, 0, 0, 0, 0]):
        self.N = 50
        self.dt = 1/self.N
        self.x0 = x0
        self.xT = xT
        self.Q = csd.diag([1, 1, 1, 1, 1, 1])
        self.R = csd.diag([0.1, 0.1])

        self.dynamics = PlanarDroneDynamics()
        self.Fmax = 3*self.dynamics.m*self.dynamics.g/2
        self.L = (self.xT - self.dynamics.x).T @ self.Q @ (self.xT - self.dynamics.x) + \
                    self.dynamics.u.T @ self.R @ self.dynamics.u
        
        self.define_integrator()
        self.define_ocp()
        self.define_solver()
    
    def define_integrator(self):
        """ An integrator for free time OCP"""
        T = csd.MX.sym('T')
        x_tilde = csd.vertcat(self.dynamics.x, T)
        ode_tilde = csd.vertcat(T*self.dynamics.ode, 0)
        
        dae = {'x': x_tilde, 'p': self.dynamics.u, 'ode': ode_tilde, 'quad': self.L}
        opts = {'tf': self.dt, 'number_of_finite_elements': 4}
        self.integrator = csd.integrator("integrator", "rk", dae, opts)

    def define_ocp(self):
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # descion variables for initial state
        xk = csd.MX.sym('xk_0', 7)

        # Initial constraints
        w += [xk]
        w0 += [self.x0, 1]
        lbw += [0, 0, -csd.inf, -csd.inf, -csd.inf, -csd.inf, -csd.inf]
        ubw += [10, 10, csd.inf, csd.inf, csd.inf, csd.inf, csd.inf]
        g += [self.x0 - xk[0:6]]
        lbg += [0, 0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0, 0]

        for k in range(self.N):
            uk = csd.MX.sym('uk_'+str(k), 2)

            w += [uk]
            w0 += [0, 0]
            lbw += [0, 0]
            ubw += [self.Fmax, self.Fmax]

            Fnext = self.integrator(x0=xk, p=uk)

            xk = csd.MX.sym('xk_'+str(k+1), 7)
            w += [xk]
            w0 += [self.x0, 1]
            lbw += [0, 0, -csd.inf, -csd.inf, -csd.inf, -csd.inf, 0.]
            ubw += [10, 10, csd.inf, csd.inf, csd.inf, csd.inf, 15.]

            J += Fnext['qf']
            g += [Fnext['xf'] - xk]
            lbg += [0, 0, 0, 0, 0, 0, 0]
            ubg += [0, 0, 0, 0, 0, 0, 0]
        
        # terminal state condition
        g += [self.xT - xk[0:6]]
        lbg += [0, 0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0, 0]

        self.ocp = OptimalControlProblem(J, csd.vertcat(*w), csd.vertcat(*g), csd.vertcat(*w0), lbw, ubw, lbg, ubg)

    def define_solver(self):
        prob = {'f': self.ocp.J, 'x': self.ocp.w, 'g': self.ocp.g}
        opts = {"expand": False,
                "verbose": False,
                "print_time": True,
                "error_on_fail": True,
                "ipopt": {"linear_solver": "mumps",
                          "max_iter": 1000,
                          'print_level': 5,
                          'sb': 'yes',  # Suppress IPOPT banner
                          'tol': 1e-9,
                  # 'warm_start_init_point': 'yes',
                  # 'warm_start_bound_push': 1e-8,
                  # 'warm_start_mult_bound_push': 1e-8,
                  # 'mu_init': 1e-5,
                  # 'hessian_approximation': 'limited-memory'
                  }}
        # opts = {}
        self.solver = csd.nlpsol('solver', 'ipopt', prob, opts)

    def solve_ocp(self):
        sol = self.solver(x0=self.ocp.w0, lbx=self.ocp.lbw, ubx=self.ocp.ubw, lbg=self.ocp.lbg, ubg=self.ocp.ubg)
        w_opt = sol['x'].full().flatten()
        px_opt = w_opt[0::9]
        py_opt = w_opt[1::9]
        phi_opt = w_opt[2::9]
        vx_opt = w_opt[3::9]
        vy_opt = w_opt[4::9]
        phidot_opt = w_opt[5::9]
        T_opt = w_opt[6]
        F1_opt = w_opt[7::9]
        F2_opt = w_opt[8::9]

        x_opt = np.array([px_opt, py_opt, phi_opt, vx_opt, vy_opt, phidot_opt])
        u_opt = np.array([F1_opt, F2_opt])
        return x_opt, u_opt, T_opt


def plot_multiple_states(x, y, phi, L):
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



if __name__ == "__main__":
    ocp = PlanarDroneOCP()
    x0 = [5, 5, 0, 0, 0, 0, 1]
    u = [ocp.dynamics.g*ocp.dynamics.m/2, ocp.dynamics.g*ocp.dynamics.m/2]

    sol = ocp.integrator(x0=x0, p=u)
    print(sol['xf'])

    x_opt, u_opt, T_opt = ocp.solve_ocp()

    tgrid = [T_opt/ocp.N*k for k in range(ocp.N+1)]

    # import matplotlib.pyplot as plt
    plt.figure(1)
    plt.clf()
    plt.plot(tgrid, x_opt[0,:], '--')
    plt.plot(tgrid, x_opt[1,:], '-')
    plt.step(tgrid, csd.vertcat(csd.DM.nan(1), u_opt[0,:]), '-.')
    plt.step(tgrid, csd.vertcat(csd.DM.nan(1), u_opt[1,:]), '-.')
    plt.xlabel('t')
    plt.legend(['px', 'py', 'u'])
    plt.grid()
    plt.show()

    print(x_opt[0,::5])
    plot_multiple_states(x_opt[0,::5], x_opt[1,::5], x_opt[2,::5], ocp.dynamics.L)