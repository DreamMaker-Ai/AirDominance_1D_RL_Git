from gym.envs.registration import register

register(
    id='myenv-v2',
    entry_point='myenv_2.envs:MyEnv'
)
