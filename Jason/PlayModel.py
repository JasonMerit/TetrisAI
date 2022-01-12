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
env = Tetris([], False, False)
clock = pygame.time.Clock()


height = 16
width = 10
screen_size = 600
black = (34, 34, 34)
grey = (184, 184, 184)
cell_size = 25
top_left_y = screen_size / 2 - height * cell_size / 2
top_left_x = screen_size / 2 - width * cell_size / 2

pygame.font.init()  # init font
STAT_FONT = pygame.font.SysFont("comicsans", 35)
AXIS_FONT = pygame.font.SysFont("comicsans", 20)


screen = pygame.display.set_mode([screen_size, screen_size])
pygame.display.set_caption('Tetris')
background = pygame.Surface(screen.get_size())

def render():
    screen.fill(black)

    # Get and draw grid
    grid = env.get_grid()
    background = (top_left_x - 1,
                  top_left_y - 1,
                  width * cell_size + 1,
                  height * cell_size + 1)
    pygame.draw.rect(screen, grey, background)


    for i in range(width):
        for j in range(height):
            val = grid[j, i]
            color = grey if val != 0 else black
            square = (top_left_x + cell_size * i,
                      top_left_y + cell_size * j,
                      cell_size - 1, cell_size - 1)
            pygame.draw.rect(screen, color, square)

    # Draw piece
    size = len(env.piece.shape[0])
    for i in range(size):
        for j in range(size):
            if env.piece.shape[i, j] == 0:
                continue
            square = (top_left_x + cell_size * (env.piece.x + j - 3),  # POSITION HERE
                      top_left_y + cell_size * (env.piece.y + i - 2),
                      cell_size, cell_size)
            pygame.draw.rect(screen, env.piece.color, square)

    # Draw "pieces placed"
    score_label = AXIS_FONT.render("Pieces Placed",1,(255,255,255))
    screen.blit(score_label, (screen_size - score_label.get_width() - 25, 120))

    # Draw lines cleared
    score_label = STAT_FONT.render(str(env.pieces_placed),1,(255,255,255))
    screen.blit(score_label, (screen_size - score_label.get_width() - 70, 150))
    
    # Draw "Highscore"
    score_label = AXIS_FONT.render("Highscore",1,(255,255,255))
    screen.blit(score_label, (screen_size - score_label.get_width() - 35, 50))
    
    # Draw highscore
    score_label = STAT_FONT.render(str(env.highscore),1,(255,255,255))
    screen.blit(score_label, (screen_size - score_label.get_width() - 70, 80))

    pygame.display.flip()

# Definitions and default settings
run = True
action_taken = False
slow = True

done = False

while run:
    #if rendering:
        #clock.tick(40)
    

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
        render()
    
env.close()





