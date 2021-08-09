from casadi import vertcat

import ocp
from plot_utils import visualize_trajectory, plot_controls, plot_pose


# define and solve OCP for obstacle-free case
# ocp_1 = ocp.PlanarDroneOCP(stage_cost_type="time")
# x_opt_1, u_opt_1, T_opt_1 = ocp_1.solve_ocp()
# tgrid_1 = [T_opt_1/ocp_1.N*k for k in range(ocp_1.N+1)]

# print(T_opt_1)
# plot_controls(tgrid_1, u_opt_1, True)
# plot_pose(tgrid_1, x_opt_1[0:3,:])
# visualize_trajectory(x_opt_1[0,::4], x_opt_1[1,::4], x_opt_1[2,::4], ocp_1.dynamics.L, save_fig=True)


# define and solve OCP with obstacles
r_obst1 = 1.5
r_obst2 = 0.6
r_obst3 = 1
r_obst4 = 0.9
p_obst1 = vertcat(5, 4)
p_obst2 = vertcat(8, 8)
p_obst3 = vertcat(2.5, 2.5)
p_obst4 = vertcat(5, 8)

obstacles = ([p_obst1, r_obst1], [p_obst2, r_obst2], [p_obst3, r_obst3], [p_obst4, r_obst4])

ocp_2 = ocp.PlanarDroneOCP(stage_cost_type="time", obstacles=obstacles)
x_opt_2, u_opt_2, T_opt_2 = ocp_2.solve_ocp()
tgrid_2 = [T_opt_2/ocp_2.N*k for k in range(ocp_2.N+1)]

print(T_opt_2)
plot_controls(tgrid_2, u_opt_2, True)
plot_pose(tgrid_2, x_opt_2[0:3,:])
visualize_trajectory(x_opt_2[0,::4], x_opt_2[1,::4], x_opt_2[2,::4], ocp_2.dynamics.L, obstacles, True)


