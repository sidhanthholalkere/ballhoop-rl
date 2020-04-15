import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np
from gym_ballhoop.envs import update, params, transition

class BallHoopEnv(gym.Env):
    """
    A ball and hoop environment.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50,
        }

    def __init__(self):

        self.viewer = None
        self.state = None
        self.reached_top = False
        self.passed_horiz_half = False
        self.passed_vert_half = False

        self.action_space = spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float64)
        self.observation_space = spaces.Box(low=np.array([-1000, -1000, -1000, -1000, -0.0001, -1000, -1000, -1000, 0]), high=np.array([1000, 1000, 1000, 1000, params.Ro + 0.001, 1000, 1000, 1000, 2]), dtype=np.float64)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        for _ in range(20):
            self.state = update.update_all(self.state, action)     
        psi = np.radians(self.state[2]) - np.pi/2
        r = self.state[4]

        reward = (r * (np.sin(psi) + 1))**3

        reward += params.Ro - params.Rb - r

        if np.sin(psi) > 0:
            self.passed_vert_half = True

        done = False
        if self.passed_vert_half and np.abs(psi - np.pi/2) < 0.1:
            done = True
        
        return np.asarray(self.state), reward, done, {}
        
    def reset(self):
        """
        Resets the environment
        """
        self.state = np.asarray([0, 0, 0, 0, params.Ro - params.Rb, 0, 0, 0, 1], dtype=np.float64)
        return np.asarray(self.state)

    def render(self, mode='human'):
        """
        Renders the environment, which consiste of the outer hoop, ball, and a marker
        that indicates the angle of the outer hoop
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
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        
    