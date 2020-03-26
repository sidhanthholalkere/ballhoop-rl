from gym.envs.registration import register

register(
    id='ballhoop-v0',
    entry_point='gym_ballhoop.envs:BallHoopEnv',
)