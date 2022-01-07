import numpy as np
from TetEnv_States import Tetris
from KerasModel import DQN

env = Tetris()
agent = DQN(env=env)
agent.train(games=10)
state, score, done, _ = env.reset()
while True:
    action, feature = agent.take_action(state)
    state, score, done, _ = env.step(action)
    print(score)
    env.render()
    if done:
        env.reset()
