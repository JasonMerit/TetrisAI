import numpy as np
from TetEnv_States import Tetris
from KerasModel import DQN
import time
import pygame
clock = pygame.time.Clock()


env = Tetris(rendering=True)
agent = DQN(env=env, epsilon=0)
# agent.load('SECOND_TRY_25000')
actions, Features, score, done, _ = env.reset()
run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            dos_lag = 0
            if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                run = False        
            elif event.key == pygame.K_p:
                pause = True
                while pause:
                    clock.tick(40)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            quit()
                        if event.type == pygame.KEYDOWN:
                            if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                                pygame.quit()
                                quit()
                            elif event.key == pygame.K_p:
                                pause = False

    action, feature = agent.take_action(actions, Features)
    actions, Features, score, done, _ = env.step(action)
    env.render()
    if done:
        actions, Features, score, done, _ = env.reset()
