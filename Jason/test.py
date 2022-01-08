# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 13:18:03 2022

@author: Jason
"""
import numpy as np
import pygame
import os
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (40,40)
S = np.array([[[0,0,0],
               [0,1,1],
               [1,1,0]],
              [[0,1,0],
               [0,1,1],
               [0,0,1]]])
x, y = 22, 10
kek = False
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
                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                 [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                 [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                 [1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

def flip():
    grid = board[2:22, 3:13]
    grid = np.flip(grid, axis = 1)
    board[2:22, 3:13] = grid

if kek:
    flip()


def get_sub_board():
    size = len(S[0])
    sub_board = board[y:y+size, x:x+size]
    return sub_board

#print(get_sub_board())

def get_grid():
    sub_board = board[2:22, 3:13]
    return sub_board

#print(get_grid())

def game_over():
    if np.any(board[1][3:13] == 1):
        return True
    return False

#print(game_over())

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
        c, lc = board[:, x], board[:, x-1]
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
    left = board[y, x-1]
    right = board[y, x+1]
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
        
        keks = sum(lc_down) + sum(cc_down) + sum(rc_down)
        holes += keks
        # print("[{}] holes: {}  ".format(x-1, keks))
    
    return holes

def negate(arr):
    # https://stackoverflow.com/questions/56594598/change-1s-to-0-and-0s-to-1-in-numpy-array-without-looping
    return np.where((arr==0)|(arr==1), arr^1, arr)

print(holes())

if False:
    a = get_sub_board()
    shape = S[0]
    print(a+shape)
    if np.any(a+shape == 2):
        print("collision")


# Jeg vil gerne kunne omskrive board til en tuppel af koordinator
if False:
    A = np.array([[0, 0, 0], [0, 1, 1], [1, 1, 0]])
    
    indices = np.where(A == 1)
    a, b = indices[0],indices[1]
    x, y = 5, 10
    a += y
    b += x
    coor = zip(a,b)
    
    for c in coor:
        board[c] = 1
    
    print(board)

if False:
    x, y = 2, 3
    size = len(S[0])
    for i in np.arange(size) + x:
        for j in np.arange(size) + y:
            print(i,j)
            
screen_size = 600
black = (34, 34, 34)
grey = (184, 184, 184)
cell_size = 25
height = 20
width = 10
top_left_y = screen_size / 2 - height * cell_size / 2
top_left_x = screen_size / 2 - width * cell_size / 2

pygame.font.init()  # init font
STAT_FONT = pygame.font.SysFont("comicsans", 20)


screen = pygame.display.set_mode([screen_size, screen_size])
pygame.display.set_caption('Tetris')
background = pygame.Surface(screen.get_size())

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

# Draw circles
for x, y in circles:
    center = (top_left_x + cell_size*(x-2.5), top_left_y + cell_size*(y-1.5))
    pygame.draw.circle(screen, grey, center, 8)

# Draw axis
for y in range(height):
    string = STAT_FONT.render(str(y+2),1,(255,255,255))
    screen.blit(string, (top_left_x - string.get_width() - 10, 
                         top_left_y+cell_size*y))
for x in range(width):
    string = STAT_FONT.render(str(x+3),1,(255,255,255))
    screen.blit(string, (top_left_x + x*cell_size + 4, 
                         top_left_y + height*cell_size))
  
pygame.display.flip()

run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            dos_lag = 0
            if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                run = False
pygame.quit()
    