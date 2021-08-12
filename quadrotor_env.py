import time
import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

import casadi as c

from stable_baselines3.common.env_checker import check_env


class PlanarQuadrotorEnv(gym.Env):
    LIMITS = np.array([0, 10])
    
    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second' : 10
    }

    def __init__(self, goal_state = np.array([9., 9., 0., 0., 0., 0.]), 
                 obstacles=None, phi_range = 2*np.pi):
        self.gravity = 9.81 #: [m/s2] acceleration
        self.mass = 1 #: [kg] mass
        self.Ixx = 0.1
        self.arm_length = 0.2 # [m]
        self.arm_width = 0.02 # [m]

        self.goal_state = goal_state
        self.obstacles = obstacles

        # reward similar to LQR
        self.Q = np.diag([1, 1, 1e-2, 1e-4, 1e-4, 1e-4])
        # self.Q = csd.diag([1e+4, 1e+4, 1e-2, 1e-3, 1e-3, 1e-3])
                
        # max and min force for each motor
        self.maxF = 3/2 * self.mass * self.gravity
        self.minF = 0.

        self.dt = 0.01              # sampling time (integration time)
        self.no_intg_steps = 0      # counter of the integration steps
        self.max_intg_steps = 500   # max number of integration steps

        high = np.array([
            10.0,
            10.0,
            phi_range,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
        ])
        
        low = np.array([
            0.0,
            0.0,
            -phi_range,
            -np.finfo(np.float32).max,
            -np.finfo(np.float32).max,
            -np.finfo(np.float32).max,
        ])
        
        self.action_space = spaces.Box(
            low = np.array([self.minF, self.minF]),
            high = np.array([self.maxF, self.maxF]),
            dtype = np.float32
        )

        self.observation_space = spaces.Box(
            low,
            high,
            dtype=np.float32
        )
        
        # state
        x =  c.MX.sym('x', 6)
        px = x[0]
        py = x[1]
        phi = x[2]
        vx = x[3]
        vy = x[4]
        phidot = x[5]
        
        # Control
        u = c.MX.sym('u', 2)
        F1 = u[0]
        F2 = u[1]
        
        m = self.mass
        r_drone = self.arm_length
        Izz = self.Ixx
        g_grav = self.gravity
        # Dynamics
        ax = -1/m * (F1 + F2) * c.sin(phi)
        ay = 1/m * (F1 + F2) * c.cos(phi) - g_grav
        phiddot = 1/Izz * (F1 - F2) * r_drone
        ode = c.vertcat(vx, vy, phidot, ax, ay, phiddot)
        
        dae = {'x': x, 'p': u, 'ode': ode}
        
        opts = {'tf': self.dt, 'number_of_finite_elements': 3}
        self.integrator = c.integrator("integrator", "rk", dae, opts)


        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        # assert self.action_space.contains(action), err_msg
            
        # action = np.clip(action, self.minF, self.maxF)
        Intg = self.integrator(x0=self.state, p=action)
        self.state = Intg['xf'].full()[:,0]        
        self.no_intg_steps += 1

        # check for collision with obstacle
        collision_with_obstacle = False
        if self.obstacles is not None:
            pos = self.state[0:2]
            for obst in self.obstacles:
                p_obst = obst[0]
                r_obst = obst[1]
                dpos = pos - p_obst
                if dpos.T @ dpos <= (self.arm_length + r_obst)**2:
                    collision_with_obstacle = True
                    break

        
        out_of_bounds = bool(
                self.state[0] >= 10. or
                self.state[0] <= 0. or
                self.state[1] >= 10. or
                self.state[1] <= 0.)           
        
        if out_of_bounds or collision_with_obstacle:
            reward = -1e+4
        else:
            dist_to_goal_reward = - (self.state - self.goal_state).T @ self.Q @ \
                                    (self.state - self.goal_state)
            # dist_to_goal_reward = 0
            being_alive_reward = 0
            reward =  being_alive_reward + dist_to_goal_reward

        done = out_of_bounds or bool(self.no_intg_steps >= self.max_intg_steps) or collision_with_obstacle
        return self.state, reward, done, {} 


    def reset(self):
        self.state = np.array([self.np_random.uniform(low=0, high=10),
                               self.np_random.uniform(low=0, high=10),
                               self.np_random.uniform(low=-c.pi/4, high=c.pi/4), 
                               0., 0., 0.])
        self.no_intg_steps = 0
        return self.state

    def set_state(self, state):
        self.state = state

    @staticmethod
    def rot_z(x0, angle, xb):
        T = np.array([ [np.cos(angle), -np.sin(angle)],
                       [np.sin(angle),  np.cos(angle)] ])
        return x0 + T.dot(xb)

    def render(self, mode='human', close=False):
        from gym.envs.classic_control import rendering
        screen_width = 800
        screen_height = 800

        x, z, phi = self.state[0:3].tolist()

        t1_xy = self.rot_z(self.state[0:2],
                            self.state[2],
                            np.array([self.arm_length, 0]))
        t2_xy = self.rot_z(self.state[0:2],
                            self.state[2],
                            np.array([-self.arm_length, 0]))

        to_xy = self.goal_state[0:2]
        
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(self.LIMITS[0], self.LIMITS[1],
                                   self.LIMITS[0], self.LIMITS[1])
            
            l,r,t,b = -self.arm_length, self.arm_length, self.arm_width, -self.arm_width
            self.frame_trans = rendering.Transform(rotation=phi, translation=(x,z))
            frame = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            frame.set_color(0, .8, .8)
            frame.add_attr(self.frame_trans)
            self.viewer.add_geom(frame)

            self.t1_trans = rendering.Transform(translation=t1_xy)
            thruster1 = self.viewer.draw_circle(.04)
            thruster1.set_color(.8, .8, 0)
            thruster1.add_attr(self.t1_trans)
            self.viewer.add_geom(thruster1)

            self.t2_trans = rendering.Transform(translation=t2_xy)
            thruster2 = self.viewer.draw_circle(.04)
            thruster2.set_color(.8, .8, 0)
            thruster2.add_attr(self.t2_trans)
            self.viewer.add_geom(thruster2)

            self.to_trans = rendering.Transform(translation=to_xy)
            objective = self.viewer.draw_circle(.02)
            objective.set_color(1., .01, .01)
            objective.add_attr(self.to_trans)
            self.viewer.add_geom(objective)

            if self.obstacles is not None:
                for obst in self.obstacles:
                    p_obst = obst[0]
                    r_obst = obst[1]
                    self.viewer.draw_circle(r_obst).add_attr(rendering.Transform(p_obst))

        self.frame_trans.set_translation(x,z)
        self.frame_trans.set_rotation(phi)
        
        self.t1_trans.set_translation(t1_xy[0], t1_xy[1])
        self.t2_trans.set_translation(t2_xy[0], t2_xy[1])
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')        

    def close(self):
        if self.viewer: self.viewer.close()
            
           


if __name__ == "__main__":
    # create an enviroment
    env = PlanarQuadrotorEnv()

    # check environment
    check_env(env)

    # create an environment with obstacles
    r_obst1 = 1.5
    r_obst2 = 0.6
    r_obst3 = 1
    r_obst4 = 0.9
    p_obst1 = np.array([5, 4])
    p_obst2 = np.array([8, 8])
    p_obst3 = np.array([2.5, 2.5])
    p_obst4 = np.array([5, 8]) 

    obstacles = ([p_obst1, r_obst1], [p_obst2, r_obst2])
    
    env2 = PlanarQuadrotorEnv(obstacles=obstacles)
    check_env(env2)

    for obst in obstacles:
        print(obst)
        
    env2.reset()
    env2.render()
    time.sleep(10)

    
    