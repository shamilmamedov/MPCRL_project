import casadi as csd
import numpy as np
from quad_dynamics import PlanarDroneDynamics


class OptimalControlProblem:
    def __init__(self, J, w, g, w0, lbw, ubw, lbg, ubg, p=None):
        self.J = J
        self.w = w
        self.g = g
        self.w0 = w0
        self.lbw = lbw
        self.ubw = ubw
        self.lbg = lbg
        self.ubg = ubg
        self.p = p


class PlanarDroneFreeTimeOCP:
    def __init__(self, x0 = [1, 1, 0, 0, 0, 0], xT = [9, 9, 0, 0, 0, 0], stage_cost_type = "quadratic",
                 obstacles = None):
        self.N = 50
        self.dt = 1/self.N
        self.x0 = x0
        self.xT = xT
        self.stage_cost_type = stage_cost_type
        self.obstacles = obstacles
        self.Q = csd.diag([1, 1, 1, 1, 1, 1])
        self.R = csd.diag([0.1, 0.1])


        self.dynamics = PlanarDroneDynamics()
        self.Fmin = 0
        self.Fmax = 3*self.dynamics.m*self.dynamics.g/2
        self.Tmax = 20.
        
        self.define_integrator()
        self.define_ocp()
        self.define_solver()
    
    def define_integrator(self):
        """ An integrator for free time OCP"""
        T = csd.MX.sym('T')
        x_tilde = csd.vertcat(self.dynamics.x, T)
        ode_tilde = csd.vertcat(T*self.dynamics.ode, 0)

        if self.stage_cost_type == "quadratic":
            self.L = (self.xT - self.dynamics.x).T @ self.Q @ (self.xT - self.dynamics.x) + \
                        self.dynamics.u.T @ self.R @ self.dynamics.u
        elif self.stage_cost_type == "time":
            self.L  = T
        
        dae = {'x': x_tilde, 'p': self.dynamics.u, 'ode': ode_tilde, 'quad': self.L}
        opts = {'tf': self.dt, 'number_of_finite_elements': 4}
        self.integrator = csd.integrator("integrator", "rk", dae, opts)

    def formulate_initial_guess(self):
        px_init = csd.linspace(self.x0[0], self.xT[0], self.N+1)
        py_init = csd.linspace(self.x0[1], self.xT[1], self.N+1)
        phi_init = csd.DM.zeros(self.N+1)
        vx_init = csd.DM.zeros(self.N+1)
        vy_init = csd.DM.zeros(self.N+1)
        phidot_init = csd.DM.zeros(self.N+1)
        x0 = csd.vertcat(px_init, py_init, phi_init, vx_init, vy_init, phidot_init)
        x0 = csd.transpose(csd.reshape(x0, (self.N+1, 6)))

        F1_init = csd.DM.ones(self.N) * self.dynamics.m*self.dynamics.g/2
        F2_init = csd.DM.ones(self.N) * self.dynamics.m*self.dynamics.g/2
        u0 = csd.vertcat(F1_init, F2_init)
        u0 = csd.transpose(csd.reshape(u0, (self.N, 2)))

        T0 = 5
        return x0, u0, T0

    def define_ocp(self):
        # define lists for optimzation variables, constraints and bounds
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        x0, u0, T0 = self.formulate_initial_guess()

        # descion variables for initial state
        xk = csd.MX.sym('xk_0', 7)

        # Initial constraints
        w += [xk]
        w0 += [x0[:,0], T0]
        lbw += [0, 0, -csd.inf, -csd.inf, -csd.inf, -csd.inf, 0.]
        ubw += [10, 10, csd.inf, csd.inf, csd.inf, csd.inf, self.Tmax]
        g += [self.x0 - xk[0:6]]
        lbg += [0, 0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0, 0]

        for k in range(self.N):
            uk = csd.MX.sym('uk_'+str(k), 2)

            w += [uk]
            w0 += [u0[:,k]]
            lbw += [self.Fmin, self.Fmin]
            ubw += [self.Fmax, self.Fmax]

            Fnext = self.integrator(x0=xk, p=uk)

            xk = csd.MX.sym('xk_'+str(k+1), 7)
            w += [xk]
            w0 += [x0[:,k+1], 1]
            lbw += [0, 0, -csd.inf, -csd.inf, -csd.inf, -csd.inf, 0.]
            ubw += [10, 10, csd.inf, csd.inf, csd.inf, csd.inf, self.Tmax]

            J += Fnext['qf']
            g += [Fnext['xf'] - xk]
            lbg += [0, 0, 0, 0, 0, 0, 0]
            ubg += [0, 0, 0, 0, 0, 0, 0]

            if self.obstacles != None:
                # Collision avoidance constraint
                pos = Fnext['xf'][0:2]
                for obst in self.obstacles:
                    p_obst = obst[0]
                    r_obst = obst[1]
                    dpos = pos - p_obst
                    g += [dpos.T @ dpos - (self.dynamics.L + r_obst)**2]
                    lbg += [0]
                    ubg += [csd.inf]
                    
        # terminal state constraint
        if self.stage_cost_type == "time":
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



class PlanarDroneFixedTimeOCP:
    def __init__(self, x0 = [1, 1, 0, 0, 0, 0], xT = [9, 9, 0, 0, 0, 0], dt = 0.01, N = 50,
                 stage_cost_type = "quadratic", obstacles = None):
        self.N = N
        self.dt = dt
        self.x0 = x0
        self.xT = xT
        self.stage_cost_type = stage_cost_type
        self.obstacles = obstacles
        self.Q = csd.diag([1e+4, 1e+4, 1e-2, 1e-3, 1e-3, 1e-3])
        self.R = csd.diag([1, 1])
        
        self.dynamics = PlanarDroneDynamics()
        self.L = (self.xT - self.dynamics.x).T @ self.Q @ (self.xT - self.dynamics.x) + \
                        self.dynamics.u.T @ self.R @ self.dynamics.u
        self.Fmin = 0
        self.Fmax = 3*self.dynamics.m*self.dynamics.g/2
        
        self.define_integrator()
        self.define_ocp()
        self.define_solver()
    
    def define_integrator(self):
        """ An integrator for fixed time OCP"""
        dae = {'x': self.dynamics.x, 'p': self.dynamics.u, 'ode': self.dynamics.ode, 'quad': self.L}
        opts = {'tf': self.dt, 'number_of_finite_elements': 5}
        self.integrator = csd.integrator("integrator", "rk", dae, opts)

    def formulate_initial_guess(self):
        px_init = csd.linspace(self.x0[0], self.xT[0], self.N+1)
        py_init = csd.linspace(self.x0[1], self.xT[1], self.N+1)
        phi_init = csd.DM.zeros(self.N+1)
        vx_init = csd.DM.zeros(self.N+1)
        vy_init = csd.DM.zeros(self.N+1)
        phidot_init = csd.DM.zeros(self.N+1)
        x0 = csd.vertcat(px_init, py_init, phi_init, vx_init, vy_init, phidot_init)
        x0 = csd.transpose(csd.reshape(x0, (self.N+1, 6)))

        F1_init = csd.DM.ones(self.N) * self.dynamics.m*self.dynamics.g/2
        F2_init = csd.DM.ones(self.N) * self.dynamics.m*self.dynamics.g/2
        u0 = csd.vertcat(F1_init, F2_init)
        u0 = csd.transpose(csd.reshape(u0, (self.N, 2)))

        return x0, u0

    def define_ocp(self):
        # define lists for optimzation variables, constraints and bounds
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        x0, u0 = self.formulate_initial_guess()

        # descion variables for initial state
        xk = csd.MX.sym('xk_0', 6)

        # initial state (will be a parameter of the optimization problem)
        x0_bar = csd.MX.sym('x0_bar', 6)

        # Initial constraints
        w += [xk]
        w0 += [x0[:,0]]
        lbw += [0, 0, -csd.inf, -csd.inf, -csd.inf, -csd.inf]
        ubw += [10, 10, csd.inf, csd.inf, csd.inf, csd.inf]
        g += [x0_bar - xk]
        lbg += [0, 0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0, 0]

        for k in range(self.N):
            uk = csd.MX.sym('uk_'+str(k), 2)

            w += [uk]
            w0 += [u0[:,k]]
            lbw += [self.Fmin, self.Fmin]
            ubw += [self.Fmax, self.Fmax]

            Fnext = self.integrator(x0=xk, p=uk)

            xk = csd.MX.sym('xk_'+str(k+1), 6)
            w += [xk]
            w0 += [x0[:,k+1]]
            lbw += [0, 0, -csd.inf, -csd.inf, -csd.inf, -csd.inf]
            ubw += [10, 10, csd.inf, csd.inf, csd.inf, csd.inf]

            J += Fnext['qf']
            g += [Fnext['xf'] - xk]
            lbg += [0, 0, 0, 0, 0, 0]
            ubg += [0, 0, 0, 0, 0, 0]

            if self.obstacles is not None:
                # Collision avoidance constraint
                pos = Fnext['xf'][0:2]
                for obst in self.obstacles:
                    p_obst = obst[0]
                    r_obst = obst[1]
                    dpos = pos - p_obst
                    g += [dpos.T @ dpos - (self.dynamics.L + r_obst)**2]
                    lbg += [0]
                    ubg += [csd.inf]

        self.ocp = OptimalControlProblem(J, csd.vertcat(*w), csd.vertcat(*g), csd.vertcat(*w0), lbw, ubw, lbg, ubg, x0_bar)

    def define_solver(self):
        prob = {'f': self.ocp.J, 'x': self.ocp.w, 'g': self.ocp.g, 'p':self.ocp.p}
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

    def solve_ocp(self, x0_bar):
        # self.ocp.w0[0:6] = x0_bar
        sol = self.solver(x0=self.ocp.w0, lbx=self.ocp.lbw, ubx=self.ocp.ubw, lbg=self.ocp.lbg, ubg=self.ocp.ubg, p=x0_bar)
        w_opt = sol['x'].full().flatten()
        # set current solution of the current ocp as initial guess of the next OCP
        self.ocp.w0 = w_opt
        return w_opt[6:8]