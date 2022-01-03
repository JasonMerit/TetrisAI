import pygame
import random

# creating the data structure for pieces
# setting up global vars
# functions
# - create_grid
# - draw_grid
# - draw_window
# - rotating shape in main
# - setting up the main

"""
10 x 20 square grid
shapes: S, Z, I, O, J, L, T
represented in order by 0 - 6
"""

pygame.font.init()

# GLOBALS VARS
s_width = 800
s_height = 600
play_width = 300  # meaning 300 // 10 = 30 width per block
play_height = 600  # meaning 600 // 20 = 20 height per block
block_size = 30
black = (34,34,34)

top_left_x = (s_width - play_width) // 2
top_left_y = s_height - play_height


# SHAPE FORMATS

S = [['.....',
      '......',
      '..00..',
      '.00...',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]

Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]

I = [['..0..',
      '..0..',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '0000.',
      '.....',
      '.....',
      '.....']]

O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]

J = [['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....']]

L = [['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....']]

T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....']]

shapes = [S, Z, I, O, J, L, T]
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]
# index 0 - 6 represent shape


class Piece(object):
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape # The drawn shape as seen above
        self.color = shape_colors[shapes.index(shape)]
        self.rotation = 0 # Index of rotation
    
def create_piece(): # Change here
    # (Spawn position and random choice of pieces)
    return Piece(5, 2, random.choice(shapes))

def create_grid(locked_positions={}):
    # grid is a matrix of positions of colors
    grid = [[black for _ in range(10)] for _ in range(20)]

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if (j, i) in locked_positions:
                c = locked_positions[(j, i)]
                grid[i][j] = c
    return grid

def convert_shape_format(piece):
    positions = []
    format = piece.shape[piece.rotation % len(piece.shape)] # Redundant % due to input
    
    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                positions.append((piece.x + j, piece.y + i))
        
    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)
        
    return positions

def valid_space(piece, grid):
    # Create list of valid empty positions
    accepted_pos = [[(j, i) for j in range(10) if grid[i][j] == black] for i in range(20)]
    accepted_pos = [j for sub in accepted_pos for j in sub] # Rewrite to remove outer nest
    
    formatted = convert_shape_format(piece)
    
    for pos in formatted:
        if pos not in accepted_pos:
            if pos[1] > -1: # Otherwise above board
                return False
    return True

def check_lost(positions):
    for pos in positions:
        x, y = pos
        if y < 0: # WHEN DED?
            return True
    return False

def draw_text_middle(text, size, color, surface):
    font = pygame.font.SysFont('comicsans', size, bold = True)
    label = font.render(text, 1, color)
    surface.blit(label, (top_left_x + play_width/2 - (label.get_width())/2, top_left_y + play_height/2 - label.get_height()/2))
   
def draw_grid(surface, grid):
    sx = top_left_x
    sy = top_left_y
    
    for i in range(len(grid)):
        pygame.draw.line(surface, (128,128,128), (sx, sy+i*block_size), (sx+play_width, sy+i*block_size))
        for j in range(len(grid[i])):
            pygame.draw.line(surface, (128,128,128), (sx+j*block_size, sy), (sx+j*block_size, sy+play_height))    
    

def clear_rows(grid, locked):
    # Clears rows and returns score
    inc = 0
    for i in range(len(grid)-1, -1, -1): # Loop backwards
        row = grid[i]
        if black not in row: # Meaning full
            inc += 1
            ind = i # number of deleted rows
            for j in range(len(row)):
                try:
                    del locked[(j,i)] # Delete each column element
                except:
                    continue
    
    if inc > 0:
        # The lambda function is used to sort by the y-value
        for key in sorted(list(locked), key = lambda x: x[1])[::-1]:
            x, y = key
            if y < ind: # Above the deleted row
                newKey = (x, y + inc)
                locked[newKey] = locked.pop(key) # new position has same color
            
    return inc

def draw_next_shape(shape, surface):
    font = pygame.font.SysFont('comicsans', 30)
    label = font.render('Next Shape', 1, (255, 255, 255))
    
    sx = top_left_x + play_width + 50
    sy = top_left_y + play_height/2 - 100
    format = shape.shape[shape.rotation % len(shape.shape)]
    
    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                pygame.draw.rect(surface, shape.color, (sx +j*block_size, sy +i*block_size, block_size, block_size), 0)
    
    surface.blit(label, (sx + 10, sy - 30))

def update_score(nscore):
    score = get_max_score()
        
    with open('scores.txt', 'w') as f:
        if nscore > score:
            f.write(str(nscore))
        else:
            f.write(str(score))
    
def get_max_score():
    with open('scores.txt', 'r') as f:
        lines = f.readlines()
        score = int(lines[0].strip()) # Removing \n
    
    return score
        
def draw_window(surface, grid, score=0, last_score = 0):
    surface.fill(black)
    
    # Title
    font = pygame.font.SysFont('comicsans', 60)
    label = font.render('Tetris', 1, (255, 255, 255))
    surface.blit(label, (top_left_x + play_width/2 - (label.get_width()/2), 30))
    
    # Score
    font = pygame.font.SysFont('comicsans', 30)
    label = font.render('Score: {}'.format(score), 1, (255, 255, 255))
    sx = top_left_x + play_width + 50
    sy = top_left_y + play_height/2 - 100
    surface.blit(label, (sx + 25, sy - 180))
    
    # Highscore
    font = pygame.font.SysFont('comicsans', 30)
    label = font.render('High Score: {}'.format(last_score), 1, (255, 255, 255))
    sx = top_left_x - 200
    sy = top_left_y + play_height/2 - 100
    surface.blit(label, (sx + 25, sy - 180))
    
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            pygame.draw.rect(surface, grid[i][j], (top_left_x + j*block_size, top_left_y + i*block_size, block_size, block_size), 0)
    
    pygame.draw.rect(surface, (180,82,80), (top_left_x, top_left_y, play_width, play_height), 4)
    draw_grid(surface, grid)
    
def main(win):
    last_score = get_max_score()
    locked_positions = {}
    grid = create_grid(locked_positions)
    change_piece = False
    run = True
    current_piece = create_piece()
    next_piece = create_piece()
    clock = pygame.time.Clock()
    fall_time = 0
    fall_speed = 0.72
    level_time = 0
    score = 0
    
    while run:
        grid = create_grid(locked_positions)
        fall_time += clock.get_rawtime() # Time since last iteration
        level_time += clock.get_rawtime()
        clock.tick()
        
        if level_time/1000 > 5:
            level_time = 0
            if fall_speed > 0.12:
                fall_speed -= 0.005
        
        if fall_time/1000 > fall_speed: # Drop piece
            fall_time = 0
            current_piece.y += 1
            if not(valid_space(current_piece, grid)) and current_piece.y > 0:
                current_piece.y -= 1
                change_piece = True
                   
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    run = False
                if event.key == pygame.K_LEFT:
                    current_piece.x -= 1
                    if not valid_space(current_piece, grid):
                        current_piece.x += 1

                if event.key == pygame.K_RIGHT:
                    current_piece.x += 1
                    if not valid_space(current_piece, grid):
                        current_piece.x -= 1
                if event.key == pygame.K_UP:
                    # rotate shape
                    current_piece.rotation = current_piece.rotation + 1 % len(current_piece.shape)
                    if not valid_space(current_piece, grid):
                        current_piece.rotation = current_piece.rotation - 1 % len(current_piece.shape)

                if event.key == pygame.K_DOWN:
                    # move shape down
                    current_piece.y += 1
                    if not valid_space(current_piece, grid):
                        current_piece.y -= 1
                
                if event.key == pygame.K_SPACE:
                    while valid_space(current_piece, grid):
                        current_piece.y += 1
                    current_piece.y -= 1
                    change_piece = True # Instantly move on
                
        # Check if piece is done
        shape_pos = convert_shape_format(current_piece)
        
        for i in range(len(shape_pos)):
            x, y = shape_pos[i]
            if y > -1:
                grid[y][x] = current_piece.color
                
        if change_piece:
            # Store information of current piece
            # and update to new piece
            for pos in shape_pos:
                p = (pos[0], pos[1])
                locked_positions[p] = current_piece.color
            current_piece = next_piece
            next_piece = create_piece()
            change_piece = False
            # Only clear row after the current piece has been placed
            score += clear_rows(grid, locked_positions)
        
        # Draw display
        draw_window(win, grid, score, last_score)
        draw_next_shape(next_piece, win)
        pygame.display.flip() 
        
        # Determine game over
        if check_lost(locked_positions):
            draw_text_middle("YOU SUCK!", 90, (255,255,255), win)
            pygame.display.flip()
            pygame.time.delay(1500)
            run = False
            update_score(score)
    
    

def main_menu(win):
    run = True
    while run:
        win.fill(black)
        draw_text_middle("Press Any Key To Play", 60, (255,255,255), win)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    run = False
                else:
                    main(win)
    pygame.display.quit()    

win = pygame.display.set_mode((s_width, s_height))
pygame.display.set_caption('Tetris')
main_menu(win)  # start game