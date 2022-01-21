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
x, y = 5, 2#11

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
                 [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
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

env = Tetris(False, None, board, True)
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

def sandwiched(x, y, grid):
    left = grid[y, x-1]
    right = grid[y, x+1]
    return left and right

def cum_wells():
    """
    Given a well, take a sum over each cell within the well. 
    The value of the cell will be the depth w.r.t. the well. 
    E.g. a well of depth 3 will have the sum 1+2+3=6

    Start from the second column to first wall (inclusive),
    find highest full cell, check sandwich for empty cells left and down,
    """
    well = 0
    grid = env.board[2:-2, 2:-1] # Cut off floor and keep one wall on either side
    for x in range(2,12): # Second column to first wall
        # c = coumn, lc = left_column
        c, lc = grid[:, x], grid[:, x-1]
        
        top_c = np.argmax(c)
        top_lc = np.argmax(lc)
        if top_lc <= top_c: # No well for column x
            continue
    
        # Iterate down through empty left column and check for sandwich
        depth, is_full = 0, lc[top_c]
        while not is_full:
            if sandwiched(x-1, top_c + depth, grid):
                circles.append((x+1, top_c + depth+2))
                well += depth + 1 # 0 indexing
            else:
                top_c += depth + 1 # Reset to new well in same column
                depth = -1
                
            depth += 1
            is_full = lc[top_c + depth]

    return well
# print(cum_wells())




def draw():
    # Draw circles
    for x, y in circles:
        center = (env.top_left_x + env.cell_size*(x-2.5), env.top_left_y + env.cell_size*(y-1.5))
        pygame.draw.circle(env.screen, env.grey, center, 8)
    
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

def old_holes():
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
        
        print(cc_down)

        holes += sum(cc_down) #+ sum(lc_down) + sum(rc_down)

    return holes

def holes_depth_and_row_holes():
    """
    Hole is any empty space below a any full cell
    Hole depth is vertical distance of full cells above hole
    Row holes are number of rows with at least one occurrence of holes
    """
    holes = 0
    hole_depth = 0
    row_holes = set()
    grid = board[2:2 + height, 3:3 + width]

    for x in range(len(grid[0])):  # Iterate through columns
        c = grid[:, x]

        # Get relevant part of column
        top = np.argmax(c)

        c_down = c[top:]

        # Find indice and amount of holes within column
        indice = np.where(c_down == 0)[0]
        # print(indice+top+2)
        c_holes = len(indice)
        
        row_holes = row_holes.union(set(indice + top))
        if c_holes == 0:  # Zero holes
            continue
        print(x, c_holes)
        holes += c_holes
        hole_depth += sum(indice) - c_holes + 1

    return holes, hole_depth, len(row_holes)

print(holes_depth_and_row_holes())

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

#print(get_top())



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
    
def eroded_cells():
    grid =  env.board[0:2 + height, 3:3 + width]
    
    row = np.array([])
    for r in range(len(grid)):  
        if grid[r].all():
            row = np.append(row, r)
    
    # Find y-values the current piece inhabits
    indices = np.where(env.piece.shape == 1)
    ys = indices[0] + env.piece.y
    
    piece_cells = 0
    for y in ys:
        if y in row:
            piece_cells += 1
            
    print(f"rows: {len(row)}")
    print("piece_cells: {}".format(piece_cells))
    
    return piece_cells * len(row)

# print(eroded_cells())

def lock_height():
    """
    Determine vertical distance to floor for current piece.
    :return: Int
    """        
    # All pieces inhabit their center row, so only check if last row contains any
    last_row_contains = int(env.piece.shape[-1].any())
    
    if env.piece.tetromino < 5:  
        return env.height - env.piece.y - last_row_contains
    else: # Longbar and square exception
        return env.height - env.piece.y - last_row_contains - 1

# print(lock_height())

def column_transitions():
    """
    Column transitions are the number of adjacent empty and full cells
    within a column. The transition from highest solid to empty above is ignored,
    likewise the transition from bottom row to floor. 
    :return: Int
    """    
    total_transitions = 0
    grid = board[2:2 + height, 3:3 + width]
    for column in range(width):
        if grid[:, column].any():  # Skip empty columns
            top = np.argmax(grid[:, column])
            previous_square = 1
            for y in range(top, height):
                if grid[y, column] != previous_square:
                    total_transitions += 1
                    previous_square = int(not previous_square)

    return total_transitions

# print(column_transitions())

def row_transitions():
    """
    Row transitions are the number of adjacent empty and full cells
    within a row. The transitions between the wall and grid are included.
    Empty rows do not contribute to the sum.
    :return: Int
    """ 
    total_transitions = 0
    grid = env.board[2:2 + env.height, 2:4 + env.width] # Include both walls
    for index, row in enumerate(grid):
        if row[1:-1].any(): # Skip empty rows
            previous_square = 1
            for x in range(len(row)):
                if grid[index, x] != previous_square:
                    total_transitions += 1
                    previous_square = int(not previous_square)

    return total_transitions

print(row_transitions())

def full_rows(): # Fuse with full lines
    grid =  board[2:2 + height, 3:3 + width] # Hvor 0?
    
    rows = 0
    for r in grid:  
        if r.all():
            rows += 1
    
    return rows

#print(full_rows())

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
                # states = env.get_final_states()
                # cycle_states(states)
                print(row_transitions())
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
        # print(eroded_cells())
        action_taken = False
        env.render()
        draw()

pygame.quit()
