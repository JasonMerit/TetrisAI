# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 09:50:23 2022

@author: Jason
"""
import pickle
from Tetris import Tetris
import pygame

agent = pickle.load(open('best.pickle', 'rb'))
render = False
env = Tetris(render)
clock = pygame.time.Clock()

# Definitions and default settings
actions = ['left', 'right', 'up', 'down']
run = True
action_taken = False
slow = True

done = False

while run:
    if render:
        clock.tick(40)
    

    # Process input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                run = False
            elif event.key == pygame.K_r:
                env.reset()
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
    
    # Let agent determine and take next action
    input = tuple(env.get_state())
    output = agent.activate(input) # Returns a tuple of best action estimation
    action = output.index(max(output)) # Take max estimated action
    env.step(action)
    env.drop()

    if render:
        env.render()
    
env.close()





