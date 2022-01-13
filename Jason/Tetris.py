# Tetris 

import pygame

import numpy as np
import random


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


class Tetris():
    """
    Tetris class acting as enviroment.
    The game data is represented using a matrix representing the board,
    and piece objects. The board is extended out of view for easy collision
    detection, as such occationally the a submatrix is constructed.
    """

    # Colors and constant dimenstions
    black = (34, 34, 34)
    grey = (184, 184, 184)
    screen_size = 600
    cell_size = 25

    def __init__(self, board, training, rendering=True):

        # super(Tetris, self).__init__()
        # self.action_space = spaces.Discrete(6)
        # Observation space contains the board, and an extra row representing the next piece
        # self.observation_space = spaces.Box(low=0, high=1, shape=(207, 1), dtype=int)

        self.training = training
        self.height = 16
        self.width = 10
        self.board = board if len(board) != 0 else self.new_board()
        self.piece = Piece()
        self.next_piece = Piece()

        self.pieces_placed = 0
        self.highscore = 0

        self.current_score = 0
        self.score = 0
        self.current_lines = 0
        self.current_height = 0
        self.number_of_lines = 0

        # Rendering Dimensions
        self.height = len(self.board) - 4
        self.width = len(self.board[0]) - 5
        self.top_left_y = self.screen_size / 2 - self.height * self.cell_size / 2
        self.top_left_x = self.screen_size / 2 - self.width * self.cell_size / 2

        if rendering:
            pygame.font.init()  # init font
            pygame.init()
            self.TXT_FONT = pygame.font.SysFont("comicsans", 25)
            self.STAT_FONT = pygame.font.SysFont("comicsans", 35)
            self.screen = pygame.display.set_mode([self.screen_size, self.screen_size])
            pygame.display.set_caption('Tetris')
            self.background = pygame.Surface(self.screen.get_size())

    def step(self, action):
        """
        Applies the given action in the environment.
        It works by taking the action and redoing if the piece ends up
        in an invalid configuration.
        :param action: Action given to environment (String)
        :return: None
        """

        if action == "left":
            self.piece.x -= 1
            if not self.valid_position():
                self.piece.x += 1
        elif action == "right":
            self.piece.x += 1
            if not self.valid_position():
                self.piece.x -= 1
        elif action == "drop":
            self.piece.y += 1
            if not self.valid_position():
                self.piece.y -= 1
                self.new_piece()
        elif action == "up":
            self.piece.y -= 1
            if not self.valid_position():
                self.piece.y += 1
        elif action == "rotate":
            self.piece.rotate()
            if not self.valid_position():
                self.piece.rotate(False)
        elif action == "lotate":
            self.piece.rotate(False)
            if not self.valid_position():
                self.piece.rotate(True)
        elif action == "slam":
            while self.valid_position():
                self.piece.y += 1
            self.piece.y -= 1
            self.new_piece()
        elif action == "change":
            self.piece.change()

    def valid_position(self):
        """
        Returns whether the current position is valid.
        Assumes piece is not out of bounds.
        """
        # Get area of board the shape covers
        x, y = self.piece.x, self.piece.y
        size = len(self.piece.shape)
        sub_board = self.board[y:y + size, x:x + size]

        # Check for collision by summing and checking for overlap
        if np.any(self.piece.shape + sub_board > 1):
            return False
        return True

    def new_piece(self):
        """
        Registers current piece into board, creates new piece and
        returns true if game over.
        Assumes final piece.
        :return: Bool
        """
        self.pieces_placed += 1

        # Find coordinates the current piece inhabits
        indices = np.where(self.piece.shape == 1)
        a, b = indices[0], indices[1]
        a, b = a + self.piece.y, b + self.piece.x
        coords = zip(a, b)

        # Change the board accordingly
        for c in coords:
            self.board[c] = 1

        # Get new piece and clear possible lines
        self.piece = self.next_piece
        self.next_piece = Piece()
        self.clear_lines()

        # Check for game over by overlapping spawn piece
        if not self.valid_position():
            if not self.training:
                self.reset()

        return not self.valid_position()

    def new_board(self):
        board = np.zeros([self.height + 2, self.width], dtype=int)
        wall = np.ones([self.height + 2, 2])
        floor = np.ones([2, self.width + 5])
        board = np.c_[np.ones(self.height + 2), wall, board, wall]
        board = np.vstack((board, floor))
        return board.astype(int)

    def clear_lines(self):
        """
        Check and clear lines if rows are full
        """
        # Get visual part of board
        grid = self.board[2:2 + self.height, 3:3 + self.width]
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
        self.board[2:2 + self.height, 3:3 + self.width] = grid

    def render(self):
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

        # Draw "pieces placed"
        score_label = self.TXT_FONT.render("Pieces Placed", 1, (255, 255, 255))
        self.screen.blit(score_label, (self.screen_size - score_label.get_width() - 25, 150))

        # Draw lines cleared
        score_label = self.STAT_FONT.render(str(self.pieces_placed), 1, (255, 255, 255))
        self.screen.blit(score_label, (self.screen_size - score_label.get_width() - 70, 180))

        # Draw "Highscore"
        score_label = self.TXT_FONT.render("Highscore", 1, (255, 255, 255))
        self.screen.blit(score_label, (self.screen_size - score_label.get_width() - 40, 50))

        # Draw highscore
        score_label = self.STAT_FONT.render(str(self.highscore), 1, (255, 255, 255))
        self.screen.blit(score_label, (self.screen_size - score_label.get_width() - 70, 80))

        # Display
        pygame.display.flip()

    def reset(self):
        """
        Resets game by creating new board and pieces
        :return: None
        """
        self.highscore = max(self.highscore, self.pieces_placed)
        self.pieces_placed = 0
        self.board = self.new_board()
        self.piece = Piece()
        self.next_piece = Piece()
        self.shift_piece = None

    def close(self):
        """
        Close down the game
        :return: None
        """
        pygame.quit()

    # %% Exploration of board

    def get_state(self):
        """
        Returns position and rotations of current piece
        :return: Tuple
        """
        return self.piece.x, self.piece.y, self.piece.rotation

    def set_state(self, state):
        self.piece.x = state[0]
        self.piece.y = state[1]
        self.piece.rotation = state[2]
        self.piece.shape = self.piece.shapes[self.piece.tetromino][self.piece.rotation]

    def visit(self, action):
        """
        Do action, observe state and undo action.
        Returns None if state is invalid.
        :return: Tuple
        """
        initial_state = self.get_state()
        if action == "left":
            self.piece.x -= 1
            was_valid = self.valid_position()
            observation = self.get_state()
        elif action == "right":
            self.piece.x += 1
            was_valid = self.valid_position()
            observation = self.get_state()
        elif action == "drop":
            self.piece.y += 1
            was_valid = self.valid_position()
            observation = self.get_state()
        elif action == "lotate":
            self.piece.rotate(False)
            was_valid = self.valid_position()
            observation = self.get_state()
        elif action == "rotate":
            self.piece.rotate()
            was_valid = self.valid_position()
            observation = self.get_state()

        self.set_state(initial_state)
        return observation if was_valid else None

    def get_after_states(self):
        """
        Cycle through all actions and return resulting states.
        :return: Set
        """
        actions = ["left", "drop", "right", "lotate", "rotate"]
        states = set()

        for action in actions:
            observation = self.visit(action)
            if observation != None:
                states.add(observation)

        return states

    def is_final_state(self, state):
        """
        Determine if current piece must be placed
        :param state: Tuple
        :return: Bool
        """
        temp = self.get_state()
        self.set_state(state)

        self.piece.y += 1
        final = not self.valid_position()

        self.set_state(temp)
        return final

    def get_grid(self):
        return self.board[2:2 + self.height, 3:3 + self.width]

    def get_top(self):
        grid = self.get_grid()
        top = len(grid)

        for x in range(len(grid[0])):
            if not grid[:, x].any():
                continue
            column = grid[:, x]
            y = np.argmax(column)
            if y < top:
                top = y

        # Convert to board, and subtract piece range
        is_long_bar = self.piece.tetromino == 6
        top += 2
        top -= 4 if is_long_bar else 3

        return max(top, 0 if is_long_bar else 1)

    def get_final_states(self):
        """
        Using BFS return all possible final states.
        Shout out to Rune for height hack
        :return: List
        """
        first_state = self.get_state()
        top = self.get_top()
        top_state = (first_state[0], top, first_state[2])

        queue = [top_state]
        expanded = set()
        step = 0
        while queue:
            expanding_state = queue.pop(0)
            # print("[{}] {}".format(step, expanding_state))
            # print("queue: {}".format(queue))
            # print("expanded: {}".format(expanded))
            if expanding_state in expanded:
                # print("continue\n")
                step += 1
                continue

            self.set_state(expanding_state)
            after_states = self.get_after_states()
            # print("visited: {}".format(after_states))
            queue += after_states
            expanded.add(expanding_state)
            step += 1
            # print("")

        # Filter out all non-final states and convert to list
        # (I don't want to check before adding to expanded)
        final_states = [state for state in expanded if self.is_final_state(state)]

        self.set_state(first_state)  # Shouldn't have to do this
        return final_states

    def get_placed_board(self):
        """
        Return grid after placing piece
        """

        # Find coordinates the current piece inhabits
        indices = np.where(self.piece.shape == 1)
        a, b = indices[0], indices[1]
        a, b = a + self.piece.y, b + self.piece.x
        coords = zip(a, b)

        # Change the board accordingly
        board = np.copy(self.board)
        for c in coords:
            board[c] = 1

        return board

    def place_state(self, state):
        """
        Places current piece into board at given state and returns done
        :param state: Tuple
        :return: Bool
        """
        self.set_state(state)
        return self.new_piece()

    # %% Evaulation and heuristics

    def board_height(self):
        """Return the height of the board."""
        board = self.board[2:22, 3:13]
        # look for any piece in any row
        board = board.any(axis=1)
        # take to sum to determine the height of the board
        return board.sum()

    def holes(self, board):
        """
        Hole is any empty space below the top full cell on neihbours
        and current column
        """
        holes = 0
        for x in range(3, 13):  # Count within visual width
            # cc = current_column, lc = left_column, rc = right_column
            lc, cc, rc = board[:, x - 1], board[:, x], board[:, x + 1]
            top = np.argmax(cc)

            # Get relevant columns
            lc_down = lc[top:]  # same height, left and down
            cc_down = cc[top + 1:]  # below and down
            rc_down = rc[top:]  # same height, right and down

            # Revert holes to filled for easy sum
            lc_down = self.negate(lc_down)
            cc_down = self.negate(cc_down)
            rc_down = self.negate(rc_down)

            holes += sum(lc_down) + sum(cc_down) + sum(rc_down)

        return holes

    def negate(self, arr):
        # Code from stackoverflow.com
        # https://stackoverflow.com/questions/56594598/change-1s-to-0-and-0s-to-1-in-numpy-array-without-looping
        return np.where((arr == 0) | (arr == 1), arr ^ 1, arr)

    def full_lines(self, board):
        # Get visual part of board
        grid = board[2:2 + self.height, 3:3 + self.width]

        full_lines = np.sum([r.all() for r in grid])

        return full_lines

    def bumpiness(self, board):
        """
        Bumpiness: The difference in heights between neighbouring columns
        """
        grid = board[2:2 + self.height + 1, 3:3 + self.width]  # Keep one floor

        bumpiness = 0
        for x in range(self.width - 1):
            bumpiness += abs(grid[:, x].argmax() - grid[:, x + 1].argmax())

        return bumpiness

    def get_evaluations(self, states):
        """
        Return evaluations in regards to heuristcs of given states
        :param: states to be evaluated (List)
        :return: List
        """
        evaluations = []

        for state in states:
            self.set_state(state)
            board = self.get_placed_board()

            holes = self.holes(board)
            full_lines = self.full_lines(board)
            placing_height = state[1]
            bumpiness = self.bumpiness(board)

            evaluations.append((holes, full_lines, placing_height, bumpiness))

        return evaluations
