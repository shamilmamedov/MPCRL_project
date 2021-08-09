import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import casadi as c


class Drone(gym.Env):
    LIMITS = np.array([0, 10])
    
    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second' : 10
    }

    def __init__(self, goal_state = np.array([5., 5., 0., 0., 0., 0.])):
        self.gravity = 9.81 #: [m/s2] acceleration
        self.mass = 1 #: [kg] mass
        self.Ixx = 0.1
        self.arm_length = 0.2 # [m]
        self.arm_width = 0.02 # [m]

        self.goal_state = goal_state

        self.Q = np.diag([1, 1, 1, 1, 1, 1])
        self.R = np.diag([0.1, 0.1])
        
        # max and min force for each motor
        self.maxF = 3/2 * self.mass * self.gravity
        self.minF = 0
        self.maxAngle = np.pi
        self.dt = 0.01
        self.no_intg_steps = 0
        self.max_intg_steps = 500

        high = np.array([
            10.0,
            10.0,
            np.pi,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
        ])
        
        low = np.array([
            0.0,
            0.0,
            -np.pi,
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
        assert self.action_space.contains(action), err_msg
            
        # action = np.clip(action, self.minF, self.maxF)
        Intg = self.integrator(x0=self.state, p=action)
        self.state = Intg['xf'].full()[:,0]        
        self.no_intg_steps += 1
        
        out_of_bounds = bool(
                self.state[0] >= 10. or
                self.state[0] <= 0. or
                self.state[1] >= 10. or
                self.state[1] <= 0.)           
        
        if out_of_bounds:
            reward = -1000
        else:
            dist_to_goal_reward = - (np.array(self.state) - self.goal_state).T @ self.Q @ \
                                    (np.array(self.state) - self.goal_state)
            # dist_to_goal_reward = 0
            being_alive_reward = 1
            reward = dist_to_goal_reward + being_alive_reward
        
        done = out_of_bounds or bool(self.no_intg_steps >= self.max_intg_steps)
        return self.state, reward, done, {} 


    def reset(self):
        self.state = np.array([5,
                               5,
                            #    self.np_random.uniform(low=-c.pi/4, high=c.pi/4), 
                               0., 0., 0., 0.])
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

        self.frame_trans.set_translation(x,z)
        self.frame_trans.set_rotation(phi)
        
        self.t1_trans.set_translation(t1_xy[0], t1_xy[1])
        self.t2_trans.set_translation(t2_xy[0], t2_xy[1])
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')        

    def close(self):
        if self.viewer: self.viewer.close()
            
           
