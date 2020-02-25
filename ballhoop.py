"""
Ball in hoop from ________
"""

import gym
import numpy as np
import transition
import update
import params

class BallHoopEnv(gym.Env):
    """
    Description:
    
    
    Source:
        This environment corresponds to the version of the ball in double hoop problem described by Jiri Zemanek and Martin Gurtner
        
    Observation:
        Type: ____
        Num    Observation
        
    Actions:
        Type: _____
        Num    Action
        
        
    Reward:
        
        
    Starting State:
    
    """
    
    def __init__(self):
        
    def step(self, action):
        self.state = update.update_all(self.state, action)
        
        # if were done, reward ...
        
    