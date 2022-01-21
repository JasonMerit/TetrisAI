"""
Trains the agent on a specified number of games
outputting the weights at a interval
"""
from Tetris import Tetris
from KerasModel import DQN

env = Tetris(training=True)
agent = DQN(env=env, state_size=8)
agent.train(games=100000, save=500, name='DQN')
