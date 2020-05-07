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
        self.normalize_actions=True

        self.viewer = None
        self.state = None

        self.action_space = spaces.Box(low=np.array([-0.7]), high=np.array([0.7]), dtype=np.float64)
        self.observation_space = spaces.Box(low=np.array([-1000, -1000, -1000, -1000, -0.0001, -1000, -1000, -1000, 0]), 
                                            high=np.array([1000, 1000, 1000, 1000, params.Ro + 0.001, 1000, 1000, 1000, 2]), 
                                            dtype=np.float64)

        self.reached_top = False
        self.finished_loop = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, verbose=False):
        """
        One step corresponds to 20 ms.
        Expects action to be a scalar between -1 and 1
        """
        assert np.abs(action) <= 1.0

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

            if verbose:
                print(f'norm_y: {normalized_y}')
                print(self.reached_top)

            if self.reached_top and normalized_y + 1 < 0.05 and not self.finished_loop:
                self.finished_loop = True
                reward += 10

        contact_penalty = -5.0 * (1 - r/(params.Ro - params.Rb))
        height_reward = (1 + normalized_y)**4

        reward = contact_penalty + height_reward

        done = self.finished_loop

        if self.finished_loop:
            reward = (-np.abs(dpsi) + 10.0) / 50.0
            done = np.abs(dpsi) < 5
        
        return np.asarray(self.state, dtype=np.float64), reward, done, {}
        
    def reset(self):
        """
        Resets the environment
        """
        self.state = np.asarray([0, 0, 0, 0, params.Ro - params.Rb, 0, 0, 0, 1], dtype=np.float64)
        self.reached_top = False
        self.finished_loop = False
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
        
    