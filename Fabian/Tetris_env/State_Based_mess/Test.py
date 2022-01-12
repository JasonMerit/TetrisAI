from TetEnv_States import Tetris
from KerasModel import DQN

env = Tetris()

a_10000 = DQN(env=env)
a_10000.load('FIRST_TRY_100')

steps = 0
games = 0
for i in range(200):
    actions, Features, score, done, _ = env.reset()
    games += 1
    while not done:
        steps += 1
        action, feature = a_10000.take_action(actions, Features)
        actions, Features, score, done, _ = env.step(action)

print(f'5000 : {steps/games}\n10000: {steps/games}')
