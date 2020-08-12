import numpy as np
from gym_ballhoop.envs import params

def before_loop_reward(state):
    """
    Calculates the reward for the given state before
    the ball has completed a loop

    Parameters
    ----------
    state : np.ndarray
        The environment's state

    Returns
    -------
    reward : float
        The reward
    """
    psi = state[2] - np.pi/2
    dpsi = state[3]
    r = state[4]
    normalized_y = r * np.sin(psi) / (params.Ro - params.Rb)

    contact_penalty = -5.0 * (1 - r/(params.Ro - params.Rb))
    height_reward = (1 + normalized_y)**4

    reward = contact_penalty + height_reward

    return reward

def after_loop_reward(state):
    """
    Calculates the reward for the given state after
    the ball has completed a loop

    Parameters
    ----------
    state : np.ndarray
        The environment's state

    Returns
    -------
    reward : float
        The reward
    """
    dpsi = state[3]
    reward = -np.abs(dpsi)

    return reward