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
        random.seed(3)
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
    
    def set(self, state):
        self.x = state[0]
        self.y = state[1]
        self.rotation = state[2]
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
    
    # Colors and constand dimenstions
    black = (34, 34, 34)
    grey = (184, 184, 184)
    

    def __init__(self, board, rendering = True):
        pygame.init()
        
        #super(Tetris, self).__init__()
        #self.action_space = spaces.Discrete(6)
        # Observation space contains the board, and an extra row representing the next piece
        #self.observation_space = spaces.Box(low=0, high=1, shape=(207, 1), dtype=int)

        self.board = board if len(board) != 0 else self.new_board()
        self.piece = Piece()
        self.next_piece = Piece()
        self.shifted = False
        
        self.current_score = 0
        self.score = 0
        self.current_lines = 0
        self.current_height = 0
        self.number_of_lines = 0
               
        # Rendering Dimensions
        screen_size = 600
        cell_size = 25
        self.height = len(self.board) - 4
        self.width = len(self.board[0]) - 5
        self.top_left_y = screen_size / 2 - self.height * cell_size / 2
        self.top_left_x = screen_size / 2 - self.width * cell_size / 2
        
        if rendering:
            self.screen = pygame.display.set_mode([screen_size, screen_size])
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

    def board_height(self):
        """Return the height of the board."""
        board = self.board[2:22, 3:13]
        # look for any piece in any row
        board = board.any(axis=1)
        # take to sum to determine the height of the board
        return board.sum()

    def get_bumpiness(self):
        board = self.board[2:22, 3:13]
        # bumpiness is the measure of the difference in heights between neighbouring columns
        bumpiness = 0
        for i in range(9):
            bumpiness += abs(board[:, i].argmax() - board[:, i + 1].argmax())

        return bumpiness

    def get_reward(self):
        reward = 0
        # reward the change in score
        reward += self.score - self.current_score
        # greatly reward a line being cleared
        reward += (self.number_of_lines - self.current_lines) * 100
        # penalize a change in height
        penalty = self.board_height() - self.current_height
        # only apply the penalty for an increase in height (not a decrease)
        if penalty > 0:
            # punish the ai for having a bumpy board only when increasing its height
            # until I find a smarter way to calculate bumpiness dependent on placing a piece
            reward -= self.get_bumpiness()
            reward -= penalty
        # big penalty for loosing
        if not self.valid_position():
            reward -= 20
        else:
            reward += 0.01
        # update the locals
        self._current_score = self.score
        self._current_lines = self.number_of_lines
        self._current_height = self.board_height()

        return reward

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
        determines if game over
        :return: None
        """
        
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
        if not self.valid_position(): # Add training bool
            self.reset()
    
    def new_board(self):
        board = np.zeros([self.height + 2, self.width])
        wall = np.ones([self.height + 2, 2])
        floor = np.ones([2, self.width + 5])
        board = np.c_[np.ones(self.height + 2), wall, board, wall]
        board = np.vstack((board, floor))
        return board

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
        background = (self.self.top_left_x - 1,
                      self.self.top_left_y - 1,
                      self.width * self.cell_size + 1,
                      self.height * self.cell_size + 1)
        pygame.draw.rect(self.screen, self.grey, background)

        for i in range(self.width):
            for j in range(self.height):
                val = grid[j, i]
                color = self.grey if val != 0 else self.black
                square = (self.self.top_left_x + self.cell_size * i,
                          self.self.top_left_y + self.cell_size * j,
                          self.cell_size - 1, self.cell_size - 1)
                pygame.draw.rect(self.screen, color, square)

        # Draw piece
        size = len(self.piece.shape[0])
        for i in range(size):
            for j in range(size):
                if self.piece.shape[i, j] == 0:
                    continue
                square = (self.self.top_left_x + self.cell_size * (self.piece.x + j - 3),  # POSITION HERE
                          self.self.top_left_y + self.cell_size * (self.piece.y + i - 2),
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

    def close(self):
        """
        Close down the game
        :return: None
        """
        pygame.quit()
    
    def get_state(self):
        """
        Returns position and rotations of current piece
        :return: Tuple
        """
        return (self.piece.x, self.piece.y, self.piece.rotation)
    
    def set_state(self, state):
        self.piece.set(state)
    
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
    
    def get_final_states(self):
        """
        Using BFS return all possible final states.
        :return: List
        """
        first_state = self.get_state()
        queue = [first_state]
        expanded = set()
        step = 0
        while queue:
            expanding_state = queue.pop(0)
            print("[{}] {}".format(step, expanding_state))
            print("queue: {}".format(queue))
            print("expanded: {}".format(expanded))
            if expanding_state in expanded:
                print("continue\n") 
                step += 1
                continue
            
            self.set_state(expanding_state)
            after_states = self.get_after_states()
            print("visited: {}".format(after_states))
            queue += after_states
            expanded.add(expanding_state)
            step += 1
            print("")
        
        # Filter out all non-final states and convert to list
        # (I don't want to check before adding to expanded)
        final_states = [state for state in expanded if self.is_final_state(state)]
        
        self.set_state(first_state) # Shouldn't have to do this
        return final_states
        
        
            
    
    
