# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 13:18:03 2022

@author: Jason
"""
import numpy as np
import pygame
S = np.array([[[0,0,0],
               [0,1,1],
               [1,1,0]],
              [[0,1,0],
               [0,1,1],
               [0,0,1]]])
x, y = 22, 10

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
                 [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])



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
    well = 0
    for x in range(3,13):
        column = board[:, x]
        y = np.argmax(column)
        left = board[y-1, x-1]
        right = board[y-1, x+1]
        if left and right:
            print("[{},{}], left: {}, right: {}".format(x, y, left, right))
            well += 1
    return well
    

print(well_cells())

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

# Draw axis
for y in range(height):
    string = STAT_FONT.render(str(y),1,(255,255,255))
    screen.blit(string, (top_left_x - string.get_width() - 10, 
                         top_left_y+cell_size*y))
for x in range(width):
    string = STAT_FONT.render(str(x),1,(255,255,255))
    screen.blit(string, (top_left_x + x*cell_size + 2, 
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
    