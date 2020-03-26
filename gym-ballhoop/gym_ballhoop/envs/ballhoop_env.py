import gym
import numpy as np
import transition
import update
import params

class BallHoopEnv(gym.Env):
    """
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
        
    def reset(self):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
        
    