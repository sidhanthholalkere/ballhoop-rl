import gym
import numpy as np
from gym_ballhoop.envs import update, params, transition, rendering

class BallHoopEnv(gym.Env):
    """
    """
    metadata = {'render.modes': ['human']}

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
        raise NotImplementedError

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        
    