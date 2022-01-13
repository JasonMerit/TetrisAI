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

    def get_pos(self):
        # returns a tuple of four, as its main use is in the search for valid states
        return self.x, self.y, self.rotation

    def update_pos(self, pos):
        # pos is a tuple of four, as it is made to compliment get_pos
        self.x, self.y, self.rotation = pos
        self.shape = self.shapes[self.tetromino][self.rotation]


# noinspection PyTypeChecker
class Tetris:
    """
    Tetris class acting as environment.
    The game data is represented using a matrix representing the board,
    and piece objects. The board is extended out of view for easy collision
    detection, as such occasionally the a submatrix is constructed.
    """
    metadata = {'render.modes': ['human']}

    # Rendering Dimensions
    screen_size = 600
    cell_size = 25
    height = 16
    width = 10
    top_left_y = screen_size / 2 - height * cell_size / 2
    top_left_x = screen_size / 2 - width * cell_size / 2
    offset = 100

    # Colors
    black = (34, 34, 34)
    grey = (184, 184, 184)

    def __init__(self):
        # Initialize board state
        self.score = 0
        self.board = self.new_board()
        self.piece = Piece()
        self.next_piece = Piece()
        self.shifted = False

    # Functions for interaction with environment
    def step(self, action):
        """
        Places the piece according to action, then spawns a new piece
        Assumes valid action
        :param action: Action given to environment (tuple)
        :return: None
        """
        self.piece.update_pos(action)
        score = self.change_in_score()
        self.score += score
        # Place the current piece, before changing to a new one
        self.place_piece()
        self.clear_lines()
        # Change piece
        self.piece = self.next_piece
        self.next_piece = Piece()
        actions, Features = self.search_actions_features()
        game_over = False
        if len(actions) < 1 or not self.valid_position():
            game_over = True

        return actions, Features, self.score, game_over, {}

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

        score += self.height - self.lock_height()   # We assume that the AI presses down all the way

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

    def column_transitions(self):
        total_transitions = 0
        board = self.get_grid()
        for column in range(self.width):
            if board[:, column].any():
                top = np.argmax(board[:, column])
                previous_square = 1
                for row in range(top, self.height):
                    if board[row, column] != previous_square:
                        total_transitions += 1
                        previous_square = 0 if previous_square else 1

        return total_transitions

    def row_transitions(self):
        total_transitions = 0
        board = self.board[2:2 + self.height, 2:4 + self.width]
        for index, row in enumerate(board):
            if row[1:-1].any():
                previous_square = 1
                for column in range(len(row)):
                    if board[index, column] != previous_square:
                        total_transitions += 1
                        previous_square = 0 if previous_square else 1

        return total_transitions

    def get_reward(self):
        return np.array([self.aggregate_height(), self.get_bumpiness(), self.lock_height(), len(
            self.full_rows()), self.change_in_score(), self.row_transitions(), self.column_transitions()])

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

    # Possible states
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
        current_pos = self.piece.get_pos()
        append_list = [current_pos]
        visited = set(current_pos)   # The action which lead to a specific position is irrelevant
        actions = []
        Features = []

        while len(append_list) > 0:  # Search through all unique states, where piece is not placed
            loop_list = append_list  # Set the looping list to the appending list
            append_list = []
            for state in loop_list:
                for action in range(1, 6):  # 1 through 5 are valid actions
                    self.piece.update_pos(state)  # Place the piece in the state from which to explore
                    self.search_step(action)
                    pos = self.piece.get_pos()
                    if pos not in visited and self.valid_position():
                        visited.add(pos)    # Ensure that we only explore from this specific state once
                        if self.placed():  # We only want to return the final positions
                            # actions.append(pos)
                            actions.append(pos)
                            Features.append(self.get_reward())  # Calculate all the heuristics at this position
                        else:   # If not a final position, we should explore from it
                            append_list.append(pos)
        self.piece.update_pos(current_pos)
        return actions, Features
