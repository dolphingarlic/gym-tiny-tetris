from gym.envs.registration import register

register(
    id='TinyTetris-v0',
    entry_point='gym_tiny_tetris.envs:TinyTetrisEnv',
)
