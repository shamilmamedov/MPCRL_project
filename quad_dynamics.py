import casadi as csd

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

    def create_integrator(self, dt):
        dae = {'x': self.x, 'p': self.u, 'ode': self.ode}
        opts = {'tf': dt, 'number_of_finite_elements': 5}
        self.integrator = csd.integrator("integrator", "rk", dae, opts)

    def simulate(self, x0, u):
        return self.integrator(x0=x0, p=u)['xf'].full()
