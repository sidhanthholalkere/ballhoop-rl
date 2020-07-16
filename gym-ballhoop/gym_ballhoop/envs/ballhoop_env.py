import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np
from gym_ballhoop.envs import update, params, transition, reward_utils

class BallHoopEnv(gym.Env):
    """
    A simulation of the ball in double hoop introduced by Martin
    Gurtner and Jiri Zemanek in https://arxiv.org/pdf/1706.07333.pdf
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50,
        }

    def __init__(self):
        self.normalize_actions=True

        self.viewer = None
        self.state = None

        self.action_space = spaces.Box(low=np.array([-0.7]), high=np.array([0.7]), dtype=np.float64)
        self.observation_space = spaces.Box(low=np.array([-1, -1, -1000, -1, -1, -1000, 0, -100, -1, -1, -1000]), 
                                            high=np.array([1, 1, 1000, 1, 1, 1000, params.Ro, 100, 1, 1, 1000]), 
                                            dtype=np.float64)

        self.reached_top = False
        self.finished_loop = False
        self.time = 0.0
        self.stage = 0

    def seed(self, seed=None):
        """
        Sets the random seed for the model

        Parameters
        ----------
        seed : 

        Returns
        -------
        List[]

        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Calculates the state of the environment after the given action

        Parameters
        ----------
        action : Float
            Describes the torque value applied to the hoop, normalized
            to [-1, 1]
        
        Returns
        -------
        observation : np.ndarray
            Observation of the state as described in _get_obs()

        reward : float
            Reward for taking the inputted action

        done : bool
            Returns whether the environment has completed the task
        """
        assert np.abs(action) <= 1.0

        self.time += 0.2
        #self.state[9] = self.time

        if self.normalize_actions:
            action *= 0.7

        reward = 0.0

        for _ in range(20):
            self.state = np.asarray(update.update_all(self.state, action), dtype=np.float64)

            psi = self.state[2] - np.pi/2
            dpsi = self.state[3]
            r = self.state[4]
            normalized_y = r * np.sin(psi) / (params.Ro - params.Rb)
            
            #self.reached_top = False
            if 0.95 <= normalized_y <= 1.0 and not self.reached_top:
                self.reached_top = True
                reward += 10

            if self.reached_top and normalized_y + 1 < 0.05 and not self.finished_loop:
                self.finished_loop = True
                reward += 10

        done = self.finished_loop
        reward = reward_utils.before_loop_reward(self.state)

        if self.finished_loop or self.stage == 1:
            reward = reward_utils.after_loop_reward(self.state)
            done = np.abs(dpsi) < 0.3
        
        return self._get_obs(self.state), reward, done, {}

    def _get_obs(self, state):
        """
        Returns an observation based on the environment's state

        Parameters
        ----------
        state : np.ndarray
            Describes the state of the environment

        Returns
        -------
        np.ndarray
            The observation based on the environment's state
        """
        th = state[0]
        Dth = state[1]
        psi = state[2]
        Dpsi = state[3]
        r = state[4]
        Dr = state[5]
        phi = state[6]
        Dphi = state[7]
        mode = state[8]

        return np.array([np.cos(th), np.sin(th), Dth, np.cos(psi), np.sin(psi), Dpsi,
                         r, Dr, np.cos(phi), np.sin(phi), Dpsi])
        
    def reset(self):
        """
        Resets the environment

        Returns
        -------
        np.ndarray
            The observation for the environment's state
            after resetting it
        """
        self.state = np.asarray([0, 0, 0, 0, params.Ro - params.Rb, 0, 0, 0, 1], dtype=np.float64)
        self.reached_top = False
        self.finished_loop = False
        self.time = 0.0

        if self.stage == 1:
            self.state = [-2.23720303e+01, -1.28929611e+02, -6.66802761e+00, -4.28627357e+01,
                 9.02342098e-02,  0.00000000e+00, -1.51823555e+02, -8.32079590e+02,
                 1.00000000e+00]

        return self._get_obs(self.state)

    def render(self, mode='human'):
        """
        Renders the environment, which consists of the outer hoop, 
        ball, and a marker that indicates the angle of the outer hoop

        Parameters
        ----------
        mode : Str
            Describes the rendering mode

        Returns
        -------

        """
        size = 500
        # we want 225 = Ro
        scale = 225. / params.Ro

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(size, size)

            # Create the hoop as two circles
            outer_hoop_outer = rendering.make_circle(250, 100, False)
            outer_hoop_inner = rendering.make_circle(225, 100, False)
            self.hooptrans = rendering.Transform()
            outer_hoop_outer.add_attr(self.hooptrans)
            outer_hoop_inner.add_attr(self.hooptrans)
            self.viewer.add_geom(outer_hoop_outer)
            self.viewer.add_geom(outer_hoop_inner)

            # Create a marker that allows us to identify at what angle the hoop is at
            hoop_marker = rendering.make_circle(12.5, 100, True)
            self.hoop_marker_trans = rendering.Transform()
            hoop_marker.add_attr(self.hoop_marker_trans)
            self.viewer.add_geom(hoop_marker)

            # Create the ball as a circle
            ball = rendering.make_circle(int(params.Rb * scale), 100, True)
            self.balltrans = rendering.Transform()
            ball.add_attr(self.balltrans)
            self.viewer.add_geom(ball)

        if self.state is None: return None

        th = self.state[0] - np.pi/2
        r = scale * self.state[4]
        psi = self.state[2] - np.pi/2

        ballx = r * np.cos(psi)
        bally = r * np.sin(psi)

        self.balltrans.set_translation(250 + ballx, 250 + bally)
        self.hooptrans.set_translation(250, 250)

        markx = 237.5 * np.cos(th)
        marky = 237.5 * np.sin(th)

        self.hoop_marker_trans.set_translation(250 + markx, 250 + marky)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        """
        Closes the viewer
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        
    