# Grid World game

# Import libraries used for this program

import pygame
import numpy as np
import random

dic = {"Z": 1, "I": 2, "J": 5, "T": 7, "L": 9, "S": 14, "O": 19}


# random.seed(dic["L"])


class Piece():
    """
    Piece class representing a tetromino.
    """

    S = np.array([[[0, 0, 0],
                   [0, 1, 1],
                   [1, 1, 0]],
                  [[0, 1, 0],
                   [0, 1, 1],
                   [0, 0, 1]]])

    Z = np.array([[[0, 0, 0],
                   [1, 1, 0],
                   [0, 1, 1]],
                  [[0, 0, 1],
                   [0, 1, 1],
                   [0, 1, 0]]])

    T = np.array([[[0, 0, 0],
                   [1, 1, 1],
                   [0, 1, 0]],
                  [[0, 1, 0],
                   [1, 1, 0],
                   [0, 1, 0]],
                  [[0, 1, 0],
                   [1, 1, 1],
                   [0, 0, 0]],
                  [[0, 1, 0],
                   [0, 1, 1],
                   [0, 1, 0]]])

    L = np.array([[[0, 0, 0],
                   [1, 1, 1],
                   [1, 0, 0]],
                  [[1, 1, 0],
                   [0, 1, 0],
                   [0, 1, 0]],
                  [[0, 0, 1],
                   [1, 1, 1],
                   [0, 0, 0]],
                  [[0, 1, 0],
                   [0, 1, 0],
                   [0, 1, 1]]])

    J = np.array([[[0, 0, 0],
                   [1, 1, 1],
                   [0, 0, 1]],
                  [[0, 1, 0],
                   [0, 1, 0],
                   [1, 1, 0]],
                  [[1, 0, 0],
                   [1, 1, 1],
                   [0, 0, 0]],
                  [[0, 1, 1],
                   [0, 1, 0],
                   [0, 1, 0]]])

    O = np.array([[[0, 0, 0, 0],
                   [0, 1, 1, 0],
                   [0, 1, 1, 0],
                   [0, 0, 0, 0]]])

    I = np.array([[[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [1, 1, 1, 1],
                   [0, 0, 0, 0]],
                  [[0, 0, 1, 0],
                   [0, 0, 1, 0],
                   [0, 0, 1, 0],
                   [0, 0, 1, 0]]])

    shapes = [S, Z, T, L,
              J, O, I]
    shape_colors = [(0, 255, 0), (255, 0, 0), (128, 0, 128), (255, 165, 0),
                    (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    def __init__(self):
        """
        Initialize the object positioned at the top of board
        :param teromino: shape of piece (int)
        :return: None
        """
        self.x, self.y = 5, 0  # SPAWN POSITION
        self.rotation = 0
        self.tetromino = random.randint(0, 6)
        self.shape = self.shapes[self.tetromino][self.rotation]
        self.color = self.shape_colors[self.tetromino]

    def rotate(self, clockwise=True):
        dir = 1 if clockwise else -1
        num_rotations = len(self.shapes[self.tetromino])
        self.rotation = (self.rotation + dir) % num_rotations
        self.shape = self.shapes[self.tetromino][self.rotation]

    def change(self):
        num_pieces = len(self.shapes)
        self.tetromino = (self.tetromino + 1) % num_pieces
        self.rotation = 0
        self.shape = self.shapes[self.tetromino][self.rotation]
        self.color = self.shape_colors[self.tetromino]


class Tetris():
    # Rendering?
    rendering = False

    # Rendering Dimensions
    screenSize = 600
    cell_size = 25
    height = 20
    width = 10
    top_left_y = screenSize / 2 - height * cell_size / 2
    top_left_x = screenSize / 2 - width * cell_size / 2
    offset = 100

    # Colors
    yellow = (236, 226, 157)
    red = (180, 82, 80)
    cyan = (105, 194, 212)
    blue = (75, 129, 203,)
    pink = (205, 138, 206)
    orange = (211, 160, 103)
    green = (75, 129, 203)
    badColor = (192, 30, 30)
    black = (34, 34, 34)
    grey = (184, 184, 184)

    def __init__(self, state=None):
        pygame.init()
        self.reward = 0
        self.score = 0
        self.board = self.new_game()
        self.piece = Piece()
        self.next_piece = Piece()

    def step(self, action):
        # Move piece and undo if invalid move

        if action == "left":
            self.piece.x -= 1
            if not self._valid_position():
                self.piece.x += 1
        elif action == "right":
            self.piece.x += 1
            if not self._valid_position():
                self.piece.x -= 1
        elif action == "up":
            self.piece.rotate()
            if not self._valid_position():
                self.piece.rotate(False)
        elif action == "down":
            if False:
                self.piece.y += 1
                if not self._valid_position():
                    self.piece.y -= 1
                    self.new_piece()

            if True:
                self.piece.rotate(False)
                if not self._valid_position():
                    self.piece.rotate(True)
        elif action == "action":
            self.piece.change()
        # print(self.piece.x, self.piece.y)

    def _valid_position(self):
        """
        Returns whether the current position is valid.
        Assumes piece is positioned inside board.
        """
        size = len(self.piece.shape)

        # Get part of board that piece inhabits
        # sx = max(size + self.piece.x - 13, 0) # Determine if sub-board exceeds board
        # sy = max(size + self.piece.y - 23, 0) # 0 if it doesn't
        n1, n2 = np.arange(size) + self.piece.x, np.arange(size) + self.piece.y
        sub_board = board[n2[:, None], n1[None, :]]

        # Check for collision by summing and checking for 2
        collision_matrix = self.piece.shape + sub_board

        if np.any(collision_matrix == 2):
            return False
        return True

    def drop(self):
        # Let piece drop 
        self.piece.y += 1
        if not self._valid_position():
            self.piece.y -= 1
            self.new_piece()

    def new_piece(self):
        # Register current piece into board and create new piece

        # Find coordinates the current piece inhabits
        indices = np.where(self.piece.shape == 1)
        a, b = indices[0], indices[1]
        a, b = a + self.piece.y, b + self.piece.x
        coor = zip(a, b)

        # Change the board accordingly
        for c in coor:
            board[c] = 1

            # Get new piece
        self.piece = self.next_piece
        self.next_piece = Piece()

    def render(self):
        if not self.rendering:
            self.init_render()

        # Clear the screen
        self.screen.fill(self.black)

        # Draw grid
        n1, n2 = np.arange(self.width) + 2, np.arange(self.height) + 2
        grid = board[n2[:, None], n1[None, :]]  # Pluck out the visual part

        background = (self.top_left_x - 1,
                      self.top_left_y - 1,
                      self.width * self.cell_size + 1,
                      self.height * self.cell_size + 1)
        pygame.draw.rect(self.screen, self.grey, background)

        for i in range(self.width):
            for j in range(self.height):
                val = grid[j, i]
                color = self.grey if val != 0 else self.black
                square = (self.top_left_x + self.cell_size * i,
                          self.top_left_y + self.cell_size * j,
                          self.cell_size - 1, self.cell_size - 1)
                pygame.draw.rect(self.screen, color, square)

        # Draw piece
        size = len(self.piece.shape[0])
        for i in range(size):
            for j in range(size):
                if self.piece.shape[i, j] == 0:
                    continue
                square = (self.top_left_x + self.cell_size * (self.piece.x + j - 2),  # POSITION HERE
                          self.top_left_y + self.cell_size * (self.piece.y + i - 2),
                          self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.piece.color, square)

        # text = self.scorefont.render("{:}".format(self.score), True, (0,0,0))
        # self.screen.blit(text, (790-text.get_width(), 10))

        # Draw game over or you won       
        # if self.game_over(self.y, grid):
        #   msg = 'Game over!'
        #  col = self.badColor
        # text = self.bigfont.render(msg, True, col)
        # textpos = text.get_rect(centerx=self.background.get_width()/2)
        # textpos.top = 300
        # self.screen.blit(text, textpos)

        # Display
        pygame.display.flip()

    def reset(self):
        self.board = self.new_game()

    def close(self):
        pygame.quit()

    def init_render(self):
        self.screen = pygame.display.set_mode([self.screenSize, self.screenSize])
        pygame.display.set_caption('Tetris')
        self.background = pygame.Surface(self.screen.get_size())
        self.rendering = True
        self.clock = pygame.time.Clock()

        # Set up game
        self.bigfont = pygame.font.Font(None, 80)
        self.scorefont = pygame.font.Font(None, 30)

    def game_over(self, y, board):
        # if np.any(board[0]):
        # return True

        return False

    def new_game(self):
        board = np.zeros([22, 10])
        wall = np.ones([22, 2])
        floor = np.ones([2, 14])
        board = np.c_[wall, board, wall]
        board = np.vstack((board, floor))
        return board

    def get_state(self):
        return self.board


# Initialize the environment
env = Tetris()
env.reset()
board = env.get_state()

# Definitions and default settings
actions = ['left', 'right', 'up', 'down']
run = True
action_taken = False
slow = True
runai = False
render = True
done = False
drop_time = 0
drop_speed = 0.06

clock = pygame.time.Clock()

while run:
    clock.tick(40)
    drop_time += clock.get_rawtime()  # Time since last iteration (ms)

    if drop_time / 1000 > drop_speed:  # Drop piece
        drop_time = 0
        env.drop()

        # Process game events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                run = False
            if event.key == pygame.K_RIGHT:
                action, action_taken = "right", True
            if event.key == pygame.K_LEFT:
                action, action_taken = "left", True
            if event.key == pygame.K_UP:
                action, action_taken = "up", True
            if event.key == pygame.K_DOWN:
                action, action_taken = "down", True
            elif event.key == pygame.K_SPACE:
                action, action_taken = "action", True

    # AI controller
    if runai:
        pass

    # Human controller
    else:
        if action_taken:
            env.step(action)
            action_taken = False

    if render:
        env.render()

env.close()
