from TetEnv_States import Tetris
from KerasModel import DQN


env = Tetris()
agent = DQN(env=env, state_size=7)
agent.train(games=100_000, save=5_000, name='FIRST_TRY')

