import numpy as np
from TetEnv_States import Tetris
from KerasModel import DQN
import time


env = Tetris()
agent = DQN(env=env)
agent.train(games=0)
actions, Features, score, done, _ = env.reset()
while True:
    action, feature = agent.take_action(actions, Features)
    actions, Features, score, done, _ = env.step(action)
    print(score)
    env.render()
    time.sleep(1)
    if done:
        actions, Features, score, done, _ = env.reset()
