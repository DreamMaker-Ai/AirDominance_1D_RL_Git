from gym.envs.registration import register

register(
    id='myenv-v1',
    entry_point='myenv_1.envs:MyEnv'
)