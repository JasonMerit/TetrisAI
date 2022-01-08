import pygame
import numpy as np
import random
import gym
from gym import spaces

dic = {"Z": 1, "I": 2, "J": 5, "T": 7, "L": 9, "S": 14, "O": 19}


# random.seed(dic["I"])


class Piece:
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
        self.rotation = 0
        self.tetromino = random.randint(0, 6)
        self.shape = self.shapes[self.tetromino][self.rotation]
        self.color = self.shape_colors[self.tetromino]
        # Spawn position depends on tetromino
        self.y = 1 if self.tetromino < 6 else 0
        self.x = 7 if self.tetromino < 5 else 6

    def rotate(self, clockwise=True):
        """
        Rotates piece in either direction
        :param clockwise: Rotates clockwise if true else counter-clockwise (bool)
        :return: None
        """
        dir = 1 if clockwise else -1
        num_rotations = len(self.shapes[self.tetromino])
        self.rotation = (self.rotation + dir) % num_rotations
        self.shape = self.shapes[self.tetromino][self.rotation]

    def change(self):
        """
        Change current piece to another piece (for debugging)
        :return: None
        """
        num_pieces = len(self.shapes)
        self.tetromino = (self.tetromino + 1) % num_pieces
        self.rotation = 0
        self.shape = self.shapes[self.tetromino][self.rotation]
        self.color = self.shape_colors[self.tetromino]


class Tetris(gym.Env):
    """
    Tetris class acting as enviroment. 
    The game data is represented using a matrix representing the board,
    and piece objects. The board is extended out of view for easy collision
    detection, as such occationally the a submatrix is constructed. 
    """
    metadata = {'render.modes': ['human']}
    
    # Rendering Dimensions
    screen_size = 600
    cell_size = 25
    height = 20
    width = 10
    top_left_y = screen_size / 2 - height * cell_size / 2
    top_left_x = screen_size / 2 - width * cell_size / 2
    offset = 100

    # Colors
    black = (34, 34, 34)
    grey = (184, 184, 184)

    def __init__(self):
        pygame.init()
        super(Tetris, self).__init__()
        self.action_space = spaces.Discrete(7)
        # Observation space contains the board, and an extra row representing the next piece
        self.observation_space = spaces.Box(low=0, high=1, shape=(207, 1), dtype=int)
        self.current_score = 0
        self.score = 0
        self.current_lines = 0
        self.current_height = 0
        self.number_of_lines = 0
        self.board = self.new_board()
        self.piece = Piece()
        self.next_piece = Piece()
        self.shifted = False
        
        self.screen = pygame.display.set_mode([self.screen_size, self.screen_size])
        pygame.display.set_caption('Tetris')
        self.background = pygame.Surface(self.screen.get_size())

    def step(self, action):
        """
        Applies the given action in the environment, and drops the piece one spot afterwards
        It works by taking the action and redoing if the piece ends up
        in an invalid configuration.
        :param action: Action given to environment (String)
        :return: state, reward and gameover
        """
        if action == 0:
            pass
        elif action == 1:
            self.piece.x -= 1   # Move left
            if not self._valid_position():
                self.piece.x += 1
        elif action == 2:
            self.piece.x += 1   # Move right
            if not self._valid_position():
                self.piece.x -= 1
        elif action == 3:
            self.piece.y += 1   # Move down
            if not self._valid_position():
                self.piece.y -= 1
                self.new_piece()
        elif action == 4:
            self.piece.rotate()     # Rotate clockwise
            if not self._valid_position():
                self.piece.rotate(False)
        elif action == 5:
            self.piece.rotate(False)    # Rotate counter-clockwise
            if not self._valid_position():
                self.piece.rotate(True)
        elif action == 6:
            while self._valid_position():   # Full drop
                self.piece.y += 1
            self.piece.y -= 1

        placed = self.drop()
        reward = self.get_reward(placed)
        if placed:
            self.new_piece()
        return self.get_state(), reward, self.game_over(), {}

    def new_board(self):
        board = np.zeros([22, 10])
        wall = np.ones([22, 2])
        floor = np.ones([2, 15])
        board = np.c_[np.ones(22), wall, board, wall]
        board = np.vstack((board, floor))
        return board

    def lock_height(self):
        if self.piece.tetromino < 6:    # all except long bar always need only an offset of 1 or 0 relative to their y
            if self.piece.shape[2].any():
                return 20 - self.piece.y - 1
            else:
                return 20 - self.piece.y
        else:   # long bar either needs an offset of 1 or 2
            if self.piece.shape[2].any():
                return 20 - self.piece.y - 1
            else:
                return 20 - self.piece.y - 2

    def aggregate_height(self):
        """Return the height of the board."""
        board = self.board[2:22, 3:13]
        height = 0
        # Messy function, do clean up:)
        for i in range(10):
            column = board[:, i]
            if column.any():
                height += 20 - np.argmax(column)
        return height

    def get_bumpiness(self):
        board = self.board[2:22, 3:13]
        # bumpiness is the measure of the difference in heights between neighbouring columns
        bumpiness = 0
        for i in range(9):
            bumpiness += abs(board[:, i].argmax() - board[:, i + 1].argmax())

        return bumpiness

    def get_reward(self, placed):
        reward = 0
        # reward the change in score
        reward += self.score - self.current_score
        # greatly reward a line being cleared
        reward += (self.number_of_lines - self.current_lines)
        # some rewards are only relevant when the tetromino is placed
        if placed:
            # reward -= (self.aggregate_height() - self.current_height)     # - when increasing, + when decreasing
            # reward -= self.get_bumpiness() * 0.2  # even boards are best, scale to match the others, bad IMO
            reward -= (self.lock_height() - 4)    # tetrominos in the bottom 4 rows are good, punish anything higher

        # big penalty for loosing
        if self.game_over():
            reward -= 20

        # update the locals
        self.current_score = self.score
        self.current_lines = self.number_of_lines
        self.current_height = self.aggregate_height()
        return reward

    def _valid_position(self):
        """
        Returns whether the current position is valid.
        Assumes piece is positioned inside board.
        """
        # Get area of board that the shape covers
        x, y = self.piece.x, self.piece.y
        size = len(self.piece.shape)
        sub_board = self.board[y:y + size, x:x + size]

        # Check for collision by summing and checking for 2
        collision_matrix = self.piece.shape + sub_board

        if np.any(collision_matrix == 2):
            return False
        return True

    def drop(self):
        """
        Drop the piece one unit down.
        :return: None
        """
        self.piece.y += 1
        if not self._valid_position():
            self.piece.y -= 1
            return True

    def new_piece(self):
        """
        Registers current piece into board, creates new piece and
        determines if game over
        """

        # Find coordinates the current piece inhabits
        indices = np.where(self.piece.shape == 1)
        a, b = indices[0], indices[1]
        a, b = a + self.piece.y, b + self.piece.x
        coords = zip(a, b)

        # Change the board accordingly
        for c in coords:
            self.board[c] = 1

        # Get new piece
        self.piece = self.next_piece
        self.next_piece = Piece()
        self.clear_lines()

        if self.game_over():
            self.reset()

    def game_over(self):
        # Game over if piece out of screen
        if np.any(self.board[2][3:13] == 1):
            return True
        return False

    def clear_lines(self):
        """
        Check and clear lines if rows are full
        """
        # Get visual part of board
        grid = self.board[2:22, 3:13]
        idx = np.array([], dtype=int)

        # Find complete rows in reverse order
        for r in reversed(range(len(grid))):  # Count rows to remove in reverse order
            if grid[r].all():
                idx = np.append(idx, r)

        # Now clear the rows
        for c in idx:
            grid = np.delete(grid, c, 0)  # Remove the cleared row
            grid = np.vstack((np.zeros(10), grid))  # Add an empty row on top
            idx += 1  # Shift remaining clear rows a line down

        # Add final result to board
        self.board[2:22, 3:13] = grid

    def render(self, mode="human"):
        # Clear the screen
        self.screen.fill(self.black)

        # Get and draw grid
        grid = self.board[2:22, 3:13]
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
                square = (self.top_left_x + self.cell_size * (self.piece.x + j - 3),  # POSITION HERE
                          self.top_left_y + self.cell_size * (self.piece.y + i - 2),
                          self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.piece.color, square)

        # Display
        pygame.display.flip()

    def reset(self):
        """
        Resets game by creating new board and pieces
        :return: None
        """
        self.board = self.new_board()
        self.piece = Piece()
        self.next_piece = Piece()
        self.shift_piece = None

        return self.get_state()

    def close(self):
        """
        Close down the game
        :return: None
        """
        pygame.quit()

    def get_state(self):
        """
        Returns all relevant information
        :return: None
        """
        next_piece_position = np.zeros(7)
        next_piece_position[self.next_piece.tetromino] = 1
        observation = np.concatenate((self.board[2:22, 3:13].flat, next_piece_position.flat))
        return observation.reshape(207, 1)
