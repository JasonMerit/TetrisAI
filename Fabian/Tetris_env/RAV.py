import numpy as np
from TetEnv_States import Tetris
from KerasModel import DQN


env = Tetris()
agent = DQN(env=env)
agent.train(games=0)
actions, Features, score, done, _ = env.reset()

env.step((3,8,0))
print(env.placed())
env.render()

