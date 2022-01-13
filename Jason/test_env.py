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

env = Tetris([], False, False)
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
        render()
        clock.tick(1)
        env.set_state(starting_state)
        render()
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
    grid = get_grid()
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
    screen.blit(score_label, (screen_size - score_label.get_width() - 25, 150))

    # Draw lines cleared
    score_label = STAT_FONT.render(str(env.pieces_placed),1,(255,255,255))
    screen.blit(score_label, (screen_size - score_label.get_width() - 70, 180))
    
    # Draw "Highscore"
    score_label = AXIS_FONT.render("Highscore",1,(255,255,255))
    screen.blit(score_label, (screen_size - score_label.get_width() - 40, 50))
    
    # Draw highscore
    score_label = STAT_FONT.render(str(env.highscore),1,(255,255,255))
    screen.blit(score_label, (screen_size - score_label.get_width() - 70, 80))

    # Draw position
    center = (top_left_x + cell_size*(env.piece.x-2.5),
              top_left_y + cell_size*(env.piece.y-1.5))
    pygame.draw.circle(screen, (255,255,255), center, 8)

    # Draw circles
    for x, y in circles:
        center = (top_left_x + cell_size*(x-2.5), top_left_y + cell_size*(y-1.5))
        pygame.draw.circle(screen, grey, center, 8)

    # Draw axis
    for y in range(height):
        string = AXIS_FONT.render(str(y+2),1,(255,255,255))
        screen.blit(string, (top_left_x - string.get_width() - 10,
                             top_left_y+cell_size*y))
    for x in range(width):
        string = AXIS_FONT.render(str(x+3),1,(255,255,255))
        screen.blit(string, (top_left_x + x*cell_size + 4,
                             top_left_y + height*cell_size))

    pygame.display.flip()



render()
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
                            quit()
                        if event.type == pygame.KEYDOWN:
                            if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                                pygame.quit()
                                quit()
                            elif event.key == pygame.K_p:
                                pause = False

    if action_taken:
        env.step(action)

        action_taken = False

        render()

pygame.quit()
