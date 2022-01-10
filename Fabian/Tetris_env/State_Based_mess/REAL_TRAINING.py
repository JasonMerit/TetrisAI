import numpy as np
from TetEnv_States import Tetris
from KerasModel import DQN


env = Tetris()
agent = DQN(env=env, state_size=5)
agent.train(games=100_000, save=5_000, name='FIRST_TRY')

