from Tetris import Tetris
from KerasModel import DQN

env = Tetris(training=True)
agent = DQN(env=env, state_size=8)
agent.train(games=100_000, save=500, name='Mess')
