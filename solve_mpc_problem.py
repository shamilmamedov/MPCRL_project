import casadi as c
from quad_dynamics import get_dynamics

# Parameters
g = 9.81
m = 1
Fmax = 3*m*g/2


N = 50
dt = 1/N
x0 = [0, 0, 0, 0, 0, 0, 1]
xf = [1, 0, 0, 0, 0, 0, 1]


# Dyncamics
ode, x, u, L = get_dynamics()

dae = {'x': x, 'p': u, 'ode': ode, 'quad': L}
# opts = {'tf': dt, 'number_of_finite_elements': 10}
opts = {'t0': 0, 'tf': dt}
F = c.integrator("F", "cvodes", dae, opts)

# Decision variables
xk = c.MX.sym('xk', 7)

# Multiple shooting constraints
w = []
w0 = []
lbw = []
ubw = []
J = 0
g = []
lbg = []
ubg = []


# Initial constraints
w += [xk]
w0 += [x0]
lbw += [-c.inf, -c.inf, -c.inf, -c.inf, -c.inf, -c.inf, -c.inf]
ubw += [c.inf, c.inf, c.inf, c.inf, c.inf, c.inf, c.inf]
g += [x0 - xk]
lbg += [0, 0, 0, 0, 0, 0, 0]
ubg += [0, 0, 0, 0, 0, 0, 0]

for k in range(N):
    uk = c.MX.sym('uk_'+str(k), 2)

    w += [uk]
    w0 += [0, 0]
    lbw += [0, 0]
    ubw += [Fmax, Fmax]

    Fnext = F(x0=xk, p=uk)

    xk = c.MX.sym('xk_'+str(k), 7)
    w += [xk]
    w0 += [x0]
    lbw += 7*[-c.inf]
    ubw += 7*[c.inf]

    J += Fnext['qf']
    g += [Fnext['xf'] - xk]
    lbg += [0, 0, 0, 0, 0, 0, 0]
    ubg += [0, 0, 0, 0, 0, 0, 0]


prob = {'f': J, 'x': c.vertcat(*w), 'g': c.vertcat(*g)}
#
# opts = {"expand": False,
#         "verbose": False,
#         "print_time": True,
#         "error_on_fail": True,
#         "ipopt": {"linear_solver": "ma57",
#                   "max_iter": 1000,
#                   'print_level': 5,
#                   'sb': 'yes',  # Suppress IPOPT banner
#                   'tol': 1e-9,
#                   # 'warm_start_init_point': 'yes',
#                   # 'warm_start_bound_push': 1e-8,
#                   # 'warm_start_mult_bound_push': 1e-8,
#                   # 'mu_init': 1e-5,
#                   # 'hessian_approximation': 'limited-memory'
#                   }}
opts = {}

# import pdb; pdb.set_trace()

solver = c.nlpsol('solver', 'ipopt', prob)


# solve it!
sol = solver(x0=c.vertcat(*w0), lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
w_opt = sol['x'].full().flatten()
