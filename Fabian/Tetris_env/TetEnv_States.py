import pygame
import numpy as np
import random
import gym
from gym import spaces


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
        tetromino represents shape of piece (int)
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
        directory = 1 if clockwise else -1
        num_rotations = len(self.shapes[self.tetromino])
        self.rotation = (self.rotation + directory) % num_rotations
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

    def get_pos(self, action):
        # returns a tuple of four, as its main use is in the search for valid states
        return self.x, self.y, self.rotation, action

    def update_pos(self, pos):
        # pos is a tuple of four, as it is made to compliment get_pos
        self.x, self.y, self.rotation, _ = pos


# noinspection PyTypeChecker
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
    height = 10
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

    # Functions for interaction with environment
    def step(self, action):
        """
        Places the piece according to action, then spawns a new piece
        Assumes valid action
        :param action: Action given to environment (tuple)
        :return: None
        """
        self.piece.x = action[0]
        self.piece.y = action[1]
        self.piece.rotation = action[2]

        score = self.change_in_score()
        self.score += score
        # Place the current piece, before changing to a new one
        self.place_piece()
        self.clear_lines()
        self.new_piece()
        actions, Features = self.search_actions_features()

        return actions, Features, self.score, not self.valid_position(), {}

    def render(self, mode="human"):
        # Clear the screen
        self.screen.fill(self.black)

        # Get and draw grid
        grid = self.get_grid()
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
        self.score = 0

        actions, Features = self.search_actions_features()
        game_over = False
        if len(actions) == 0:
            game_over = True
        return actions, Features, self.score, game_over, {}

    def close(self):
        """
        Close down the game
        :return: None
        """
        pygame.quit()

    # Grid/Board functions
    def get_grid(self):
        grid = self.board[2:self.height + 2, 3:self.width + 3]
        return grid

    def new_board(self):
        board = np.zeros([self.height + 2, self.width])
        wall = np.ones([self.height + 2, 2])
        floor = np.ones([2, self.width + 5])
        board = np.c_[np.ones(self.height + 2), wall, board, wall]
        board = np.vstack((board, floor))
        return board

    def clear_lines(self):
        # Now clear the rows
        idx = self.full_rows()
        grid = self.get_grid()
        for c in idx:
            grid = np.delete(grid, c, 0)  # Remove the cleared row
            grid = np.vstack((np.zeros(10), grid))  # Add an empty row on top
            idx += 1  # Shift remaining clear rows a line down

        # Add final result to board
        self.board[2:self.height + 2, 3:self.width + 3] = grid

    # Piece related functions
    def valid_position(self):
        """
        Returns whether the current position is valid.
        Assumes piece is positioned inside board.
        """
        # Get the area of board that the shape covers
        x, y = self.piece.x, self.piece.y
        size = len(self.piece.shape)
        sub_board = self.board[y:y + size, x:x + size]

        # Check for collision by summing and checking for 2
        collision_matrix = self.piece.shape + sub_board
        print(collision_matrix)
        if np.any(collision_matrix > 1):
            return False
        return True

    def placed(self):
        """
        Drop the piece one unit down.
        :return True if placed
        """
        self.piece.y += 1
        if not self.valid_position():
            self.piece.y -= 1
            return True
        return False

    def place_piece(self):
        # Find coordinates the current piece inhabits
        indices = np.where(self.piece.shape == 1)
        a, b = indices[0] + self.piece.y, indices[1] + self.piece.x
        coords = zip(a, b)

        # Change the board accordingly
        for c in coords:
            self.board[c] = 1

    def remove_piece(self):
        indices = np.where(self.piece.shape == 1)
        a, b = indices[0] + self.piece.y, indices[1] + self.piece.x
        coords = zip(a, b)

        # Change the board accordingly
        for c in coords:
            self.board[c] = 0

    def new_piece(self):
        """
        Registers current piece into board, creates new piece and
        determines if game over
        """
        # Get new piece
        self.piece = self.next_piece
        self.next_piece = Piece()

        if self.game_over():
            self.reset()

    # Heuristic calculations
    def change_in_score(self):
        score = 0
        lines = self.full_rows()

        if len(lines) == 1:
            score += 40
        elif len(lines) == 2:
            score += 100
        elif len(lines) == 3:
            score += 300
        elif len(lines) == 4:
            score += 1200

        score += 20 - self.lock_height()  # The AI is assumed to press down instantly, thus triggering the reward

        return score

    def lock_height(self):
        if self.piece.tetromino < 6:  # all except long bar always need only an offset of 1 or 0 relative to their y
            if self.piece.shape[2].any():
                return self.height - self.piece.y - 1
            else:
                return self.height - self.piece.y
        else:  # long bar either needs an offset of 1 or 2
            if self.piece.shape[2].any():
                return self.height - self.piece.y - 1
            else:
                return self.height - self.piece.y - 2

    def aggregate_height(self):
        """Return the height of the board."""
        board = self.get_grid()
        height = 0
        # Messy function, do clean up:)
        for i in range(10):
            column = board[:, i]
            if column.any():
                height += 20 - np.argmax(column)
        return height

    def get_bumpiness(self):
        board = self.get_grid()
        # bumpiness is the measure of the difference in heights between neighbouring columns
        bumpiness = 0
        for i in range(9):
            bumpiness += abs(board[:, i].argmax() - board[:, i + 1].argmax())
        return bumpiness

    def full_rows(self):
        """
        Check and clear lines if rows are full
        """
        # Get visual part of board
        grid = self.get_grid()
        idx = np.array([], dtype=int)

        # Find complete rows in reverse order
        for r in reversed(range(len(grid))):  # Count rows to remove in reverse order
            if grid[r].all():
                idx = np.append(idx, r)
        return idx

    def difference_of_column_sums(self):
        column_sum = 0
        board = self.get_grid()
        for i in range(1, self.width):
            column_sum += abs(board[:, i].sum() - board[:, i - 1].sum())
        return column_sum

    def get_reward(self):
        return self.aggregate_height(), self.get_bumpiness(), self.lock_height(), len(
            self.full_rows()), self.change_in_score()

    # The SUPER function
    def get_possible_actions_and_features(self):
        """
        Checks for every possible configuration of the current piece if
        the piece is in a valid position and if it has been placed
        returns a list of [states, features]
        """
        actions = []
        Features = []
        current_piece = (self.piece.x, self.piece.y, self.piece.rotation)
        for rotation in range(len(self.piece.shapes[self.piece.tetromino])):  # Check every rotation
            self.piece.rotation = rotation
            self.piece.shape = self.piece.shapes[self.piece.tetromino][rotation]
            for x in range(3, self.width + 2):  # Check every potentially valid x value
                self.piece.x = x
                for y in range(2, self.height + 1):  # Check every potentially valid y value
                    self.piece.y = y
                    if self.valid_position() and self.placed():  # If the position is valid, and piece is placed
                        # Placing the piece each time makes calculating heuristics simpler
                        self.place_piece()
                        actions.append((x, y, rotation))
                        Features.append(self.get_reward())
                        self.remove_piece()
                        # For rendering purposes, the piece must be placed back to its spawn
        self.piece.x, self.piece.y, self.piece.rotation = current_piece[0], current_piece[1], current_piece[2]
        return actions, Features

    def search_step(self, action):
        if action == 1:
            self.piece.x -= 1  # Move left
        elif action == 2:
            self.piece.x += 1  # Move right
        elif action == 3:
            self.piece.y += 1  # Move down
        elif action == 4:
            self.piece.rotate()  # Rotate clockwise
        elif action == 5:
            self.piece.rotate(False)  # Rotate counter-clockwise

    def search_actions_features(self):
        """
        Breadth first search, using two lists for the current branches one to loop
        through and one to append to, one list for all the valid states visited
        to prevent a case of two steps forward and two steps back infinite looping.
        If a valid position where the piece is placed is found, its coordinates
        and rotation are appended to the actions list, and its features are appended
        to the Features list.

        returns actions and Features lists separately
        """
        current_pos = self.piece.get_pos(0)
        append_list = [current_pos]
        visited = [current_pos]
        actions = []
        Features = []

        while len(append_list) > 0:  # Search through all unique states, where piece is not placed
            loop_list = append_list  # Set the looping list to the appending list
            append_list = []
            for state in loop_list:
                for action in range(1, 6):
                    if action != state[3]:  # No reason to try the previous state
                        self.search_step(action)
                        pos = self.piece.get_pos(action)
                        if pos not in visited and self.valid_position():
                            print(pos, self.valid_position())
                            print(self.board)
                            visited.append(pos)
                            if self.placed():  # We only want to return the final positions
                                actions.append(
                                    pos[:-1])  # Cut out the action when appending to the actions list (ironic)
                                Features.append(self.get_reward())  # Calculate all the heuristics at this position
                            else:  #
                                append_list.append(pos)

        self.piece.update_pos(current_pos)
        return actions, Features
