import gym
import numpy as np
from gym_ballhoop.envs import update, params, transition, rendering

class BallHoopEnv(gym.Env):
    """
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 1000,
        }

    def __init__(self):

        self.viewer = None
        self.state = None

    def step(self, action):
        self.state = update.update_all(self.state, action)

        done = False

        psi = self.state[2]
        r = self.state[4]

        reward = r * (np.sin(psi) + 1)

        return self.state, reward, done, {}
        
    def reset(self):
        self.state = np.asarray([0, 0, 0, 0, params.Ro - params.Rb, 0, 0, 0, 1], dtype=np.float64)

    def render(self, mode='human'):
        size = 500
        
        # we want 225 = Ro
        scale = 225. / params.Ro

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(size, size)

            outer_hoop_outer = rendering.make_circle(250, 100, False)
            outer_hoop_inner = rendering.make_circle(225, 100, False)
            self.hooptrans = rendering.Transform()
            outer_hoop_outer.add_attr(self.hooptrans)
            outer_hoop_inner.add_attr(self.hooptrans)
            self.viewer.add_geom(outer_hoop_outer)
            self.viewer.add_geom(outer_hoop_inner)

            hoop_marker = rendering.make_circle(12.5, 100, True)
            self.hoop_marker_trans = rendering.Transform()
            hoop_marker.add_attr(self.hoop_marker_trans)
            self.viewer.add_geom(hoop_marker)

            ball = rendering.make_circle(int(params.Rb * scale), 100, True)
            self.balltrans = rendering.Transform()
            ball.add_attr(self.balltrans)
            self.viewer.add_geom(ball)

        if self.state is None: return None

        th = np.radians(self.state[0]) - np.pi/2
        r = scale * self.state[4]
        psi = np.radians(self.state[2]) - np.pi/2

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
        
    