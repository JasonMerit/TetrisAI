# -*- coding: utf-8 -*-
"""
TestingEnvironment. Draw and manipulate board to see how environment
responds. All rendering is done from this script.
"""
import numpy as np
from Tetris import Tetris
import pygame
import os
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (40,40)

h_flip = False
x, y = 3, 11

circles = []


board = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                 [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                 [1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

height = len(board) - 4
width = len(board[0]) - 5

def flip():
    grid = board[2:2 + height, 3:3 + width]
    grid = np.flip(grid, axis = 1)
    board[2:2 + height, 3:3 + width] = grid

if h_flip:
    flip()

env = Tetris(board, False, True)
env.set_state((x, y, 0))

def get_grid():
    return env.board[2:2 + height, 3:3 + width]

def well_cells():
    """
    Counting the well count by empty cells above their respective full columns
    sandwiched from sides with full cells.

    Start from the second column to first wall (inclusive),
    find highest full cell, check sandwich for empty cells left and down,
    """
    well = 0
    for x in range(4,14):
        # c = coumn, lc = left_column
        c, lc = env.board[:, x], env.board[:, x-1]
        top_c = np.argmax(c)
        top_lc = np.argmax(lc)
        if top_lc <= top_c:
            continue

        # Iterate down through empty left column and check for sandwich
        y, value = 0, lc[top_c]
        while value != 1:
            if sandwiched(x-1, top_c + y):
                circles.append((x-1, top_c + y))
                well += 1
            y += 1
            value = lc[top_c + y]

    return well

def sandwiched(x, y):
    left = env.board[y, x-1]
    right = env.board[y, x+1]
    return left and right

def draw():
    # Draw circles
    for x, y in circles:
        center = (env.top_left_x + env.cell_size*(x-2.5), env.top_left_y + env.cell_size*(y-1.5))
        env.pygame.draw.circle(env.screen, env.grey, center, 8)
    
    # Draw axis
    for y in range(height):
        string = env.TXT_FONT.render(str(y+2),1,(255,255,255))
        env.screen.blit(string, (env.top_left_x - string.get_width() - 10,
                             env.top_left_y+env.cell_size*y))
    for x in range(width):
        string = env.TXT_FONT.render(str(x+3),1,(255,255,255))
        env.screen.blit(string, (env.top_left_x + x*env.cell_size + 4,
                             env.top_left_y + env.height*env.cell_size))
    
    # Display
    pygame.display.flip()


#print(well_cells())

def holes():
    """
    Hole is any empty space below the top full cell on neihbours
    and current column
    """
    holes = 0
    for x in range(3,13): # Count within visual width
        # cc = current_column, lc = left_column, rc = right_column
        lc, cc, rc = board[:, x-1], board[:, x], board[:, x+1]
        top = np.argmax(cc)

        # Get relevant columns
        lc_down = lc[top:] #same height, left and down
        cc_down = cc[top+1:] # below and down
        rc_down = rc[top:] # same height, right and down

        # Revert holes to filled
        lc_down = negate(lc_down)
        cc_down = negate(cc_down)
        rc_down = negate(rc_down)

        holes += sum(lc_down) + sum(cc_down) + sum(rc_down)

    return holes

def negate(arr):
    # https://stackoverflow.com/questions/56594598/change-1s-to-0-and-0s-to-1-in-numpy-array-without-looping
    return np.where((arr==0)|(arr==1), arr^1, arr)


def get_top():
    grid = get_grid()
    top = len(grid) # Set to floor level

    for x in range(len(grid[0])):
        if not grid[:, x].any():
            continue
        column = grid[:, x]
        y = np.argmax(column)
        if y < top:
            top = y
    
    # Convert to board, and subtract piece range
    is_long_bar = env.piece.tetromino == 6
    top += 2
    top -= 4 if is_long_bar else 3

    return max(top, 0 if is_long_bar else 1)

print(get_top())



def cycle_states(states):
    starting_state = env.get_state()
    for state in states:
        env.set_state(state)
        env.render()
        clock.tick(1)
        env.set_state(starting_state)
        env.render()
        clock.tick(10)

def bumpiness():
    """
    Bumpiness: The difference in heights between neighbouring columns
    """
    grid =  env.board[2:2 + height + 1, 3:3 + width] # Keep one floor
    
    bumpiness = 0
    for x in range(width-1):
        bumpiness += abs(grid[:, x].argmax() - grid[:, x + 1].argmax())
        
    return bumpiness

#print(bumpiness())
    





env.render()
draw()
clock = pygame.time.Clock()
action_taken = False
run = True

while run:
    clock.tick(40)

    # Process game events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            dos_lag = 0
            if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                run = False
            if event.key == pygame.K_RIGHT:
                action, action_taken = "right", True
            if event.key == pygame.K_LEFT:
                action, action_taken = "left", True
            if event.key == pygame.K_UP and env.piece.y > 0:
                action, action_taken = "up", True
            if event.key == pygame.K_DOWN:
                action, action_taken = "drop", True
            if event.key == pygame.K_z:
                action, action_taken = "lotate", True
            elif event.key == pygame.K_x:
                action, action_taken = "rotate", True
            elif event.key == pygame.K_SPACE:
                action, action_taken = "slam", True
            elif event.key == pygame.K_e:
                action, action_taken = "change", True
            elif event.key == pygame.K_r:
                env.reset()
            elif event.key == pygame.K_t: # TEST HERE
                states = env.get_final_states()
                cycle_states(states)
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

    if action_taken:
        env.step(action)

        action_taken = False
        env.render()
        draw()

pygame.quit()
