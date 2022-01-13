# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 09:50:23 2022

@author: Jason
"""
import pickle
from Tetris import Tetris
import pygame

agent = pickle.load(open('best.pickle', 'rb'))
rendering = True
fps = 15

env = Tetris(False, [], rendering)

clock = pygame.time.Clock()




# Definitions and default settings
run = True
action_taken = False
slow = True
done = False

while run:
    if rendering:
        clock.tick(fps)
    

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
                            run = False
                            pause = False
                        if event.type == pygame.KEYDOWN:
                            if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                                pygame.quit()
                                run = False
                                pause = False
                            elif event.key == pygame.K_p:
                                pause = False
    
    # Let agent determine and take next action
    # Find all final_states and evaluate them
    states = env.get_final_states()
    evaluations = env.get_evaluations(states)
    
    # Pass the evaluation for each state into the NN
    outputs = [agent.activate(input) for input in evaluations]

    # Go to best scored state
    best_index = outputs.index(max(outputs))
    best_state = states[best_index]
    done = env.place_state(best_state) 

    if rendering:
        env.render()
    
pygame.quit()





