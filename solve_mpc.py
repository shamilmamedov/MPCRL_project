from casadi import vertcat
import matplotlib.pyplot as plt
import numpy as np


import ocp
from plot_utils import visualize_trajectory, plot_controls, plot_pose
from quad_dynamics import PlanarDroneDynamics


def solve_OCP_without_obstacles():
    ocp_1 = ocp.PlanarDroneFreeTimeOCP(stage_cost_type="time")
    x_opt_1, u_opt_1, T_opt_1 = ocp_1.solve_ocp()
    tgrid_1 = [T_opt_1/ocp_1.N*k for k in range(ocp_1.N+1)]

    print("Minimum time to reach the goal state is: ", T_opt_1)
    plot_controls(tgrid_1, u_opt_1, True)
    plot_pose(tgrid_1, x_opt_1[0:3,:])
    visualize_trajectory(x_opt_1[0,::4], x_opt_1[1,::4], x_opt_1[2,::4], ocp_1.dynamics.L, save_fig=True)


def solve_OCP_with_obstacles():
    #  define and solve OCP with obstacles
    r_obst1 = 1.5
    r_obst2 = 0.6
    r_obst3 = 1
    r_obst4 = 0.9
    p_obst1 = vertcat(5, 4)
    p_obst2 = vertcat(8, 8)
    p_obst3 = vertcat(2.5, 2.5)
    p_obst4 = vertcat(5, 8)

    obstacles = ([p_obst1, r_obst1], [p_obst2, r_obst2], [p_obst3, r_obst3], [p_obst4, r_obst4])

    ocp_2 = ocp.PlanarDroneFreeTimeOCP(stage_cost_type="time", obstacles=obstacles)
    x_opt_2, u_opt_2, T_opt_2 = ocp_2.solve_ocp()
    tgrid_2 = [T_opt_2/ocp_2.N*k for k in range(ocp_2.N+1)]

    print(T_opt_2)
    plot_controls(tgrid_2, u_opt_2, True)
    plot_pose(tgrid_2, x_opt_2[0:3,:])
    visualize_trajectory(x_opt_2[0,::4], x_opt_2[1,::4], x_opt_2[2,::4], ocp_2.dynamics.L, obstacles, True)


def solve_MPC_with_obstacles():
    dt = 0.05
    no_sim_steps = 75
    mpc_horizon = 50

    model = PlanarDroneDynamics()
    model.create_integrator(dt)

    #  define and solve OCP with obstacles
    r_obst1 = 1.5
    r_obst2 = 0.6
    r_obst3 = 1
    r_obst4 = 0.9
    p_obst1 = vertcat(5, 4)
    p_obst2 = vertcat(8, 8)
    p_obst3 = vertcat(2.5, 2.5)
    p_obst4 = vertcat(5, 8)

    obstacles = ([p_obst1, r_obst1], [p_obst2, r_obst2], [p_obst3, r_obst3], [p_obst4, r_obst4])

    ocp_solver = ocp.PlanarDroneFixedTimeOCP(dt=dt, N=mpc_horizon, obstacles=obstacles)

    x_traj = np.zeros((6, no_sim_steps+1))
    u_traj = np.zeros((2, no_sim_steps))

    x0_bar = [1, 1, 0, 0, 0, 0]
    x_traj[:,0] = x0_bar
    
    for k in range(no_sim_steps):
        x_k =  x_traj[:,k]
        u_k = ocp_solver.solve_ocp(x_k)

        # define measurement noise
        pos_noise = 0.02*np.random.normal(size=(2,))
        vel_noise = 0.02*np.random.normal(size=(2,))
        orient_noise =  0.02*np.random.normal(size=(1,))
        ang_vel_noise = 0.02*np.random.normal(size=(1,))
        noise = np.concatenate((pos_noise, orient_noise, vel_noise, ang_vel_noise))

        u_traj[:,k] = u_k
        x_traj[:,k+1] = model.simulate(x_traj[:,k], u_k).reshape((6,)) + noise


    np.save('mpc_state_trajectory.npy', x_traj)
    np.save('mpc_controls', u_traj)

    t = np.arange(0, no_sim_steps+1)
    plot_controls(t, u_traj)

    fig, ax = plt.subplots()
    ax.plot(t, x_traj[0,:], label=r'$x$')
    ax.plot(t, x_traj[1,:], label=r'$y$')
    ax.legend()
    ax.grid()

    visualize_trajectory(x_traj[0,::4], x_traj[1,::4], x_traj[2,::4], ocp_solver.dynamics.L, obstacles)

    plt.show()


# solve_MPC_with_obstacles()
t = np.arange(0, 75+1)
x_traj = np.load('mpc_state_trajectory.npy')
u_traj = np.load('mpc_controls.npy')

plot_controls(t, u_traj, save_fig=True, fig_name='controls_pr3')


#  define and solve OCP with obstacles
r_obst1 = 1.5
r_obst2 = 0.6
r_obst3 = 1
r_obst4 = 0.9
p_obst1 = vertcat(5, 4)
p_obst2 = vertcat(8, 8)
p_obst3 = vertcat(2.5, 2.5)
p_obst4 = vertcat(5, 8)

obstacles = ([p_obst1, r_obst1], [p_obst2, r_obst2], [p_obst3, r_obst3], [p_obst4, r_obst4])

visualize_trajectory(x_traj[0,::4], x_traj[1,::4], x_traj[2,::4], 0.2, obstacles, 
                    save_fig=True, fig_name='trajectory_pr3')
