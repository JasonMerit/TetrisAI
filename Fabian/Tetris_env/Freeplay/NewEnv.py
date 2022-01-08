import gym
from gym import spaces
import numpy as np
import random
import pygame


framerate = 20
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
               [1, 1, 0],
               [0, 1, 0]],
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

shapes = [S, Z, T, L, J, O, I]
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]
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


class NewEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, movement):
        self.rendering = False
        super(NewEnv, self).__init__()
        # Action space contains move sideways, rotate both directions, soft drop and do nothing
        self._tick_rate = 0
        self.action_space = spaces.Discrete(len(movement))
        # Observation space contains the board, and a top row representing the next piece
        self.observation_space = spaces.Box(low=0, high=1, shape=(23, 10), dtype=np.int)
        # Initialize scoring parameters
        self._score = 0
        self._current_score = 0
        self._current_lines = 0
        self._current_height = 0
        self._number_of_lines = 0
        self._game_over = False
        # Set board
        self.height = 25    # Three rows of ones on the left side to check for invalid long bar rotations, two on the right
        self.width = 15     # One row at the top to represent next piece, two at the bottom for collisions
        self._board = np.zeros([self.height, 10])
        a = np.c_[np.ones(self.height), np.ones(self.height), np.ones(self.height), self._board, np.ones(self.height), np.ones(self.height)]
        self._board = np.vstack((a, np.ones(self.width), np.ones(self.width)))
        # Set the piece spawning location
        self._piece_x = 4
        self._piece_y = 1
        self.framerate = framerate
        self._placed = False
        # Initialize random pieces, save their numbers for rendering
        self._current_piece_n = random.randint(0, 6)
        self._next_piece_n = random.randint(0, 6)
        self._current_piece = shapes[self._current_piece_n][0]
        self._next_piece = shapes[self._next_piece_n][0]
        self._board[0, self._next_piece_n + 2] = 1

    def step(self, action):
        reward = 0
        self._take_action(action)   # Move piece according to action
        self._tick()    # Move down every other frame

        # If the tetromino has been placed clear lines, check for game over and reset to spawn
        if self._placed:
            self._board = self._draw_piece()  # Draw piece only when placed, before changing to next piece
            self._clear_lines()
            self._placed = False
            self._piece_y = 1
            self._piece_x = 4
            # Set current piece to next and get a new random next piece
            self._current_piece = self._next_piece
            self._current_piece_n = self._next_piece_n
            self._next_piece_n = random.randint(0, 6)
            self._next_piece = shapes[self._next_piece_n][0]
            self._board[0, self._next_piece_n + 2] = 1
            if self._invalid_position():
                self._game_over = True

        reward = self._get_reward()
        return self._get_grid(True), reward, self._game_over, {}

    def _get_grid(self, Observation):
        n1, n2 = np.arange(10) + 2, np.arange(22) + 2
        grid = self._board[n2[:, None], n1[None, :]]
        # For the model it must return the top row
        if Observation:
            board = self._draw_piece()
            grid = board[n2[:, None], n1[None, :]]
            grid = np.vstack((self._board[0][2:12], grid))
        return grid

    def _draw_piece(self):
        """
        Returns the board with the current piece on it
        """
        length = len(self._current_piece)
        board = self._board.copy()
        for i, P in enumerate(self._current_piece):
            board[self._piece_y + i][self._piece_x:self._piece_x+length] = P
        return board

    def _take_action(self, action):
        # Take the chosen action (can be none):
        if action == 1:
            self.rotate(True)  # Rotate clockwise
        elif action == 2:
            self.rotate(False)  # Rotate counter-clockwise
        elif action == 3:
            self._piece_y += 1  # Move down one square, award one point for this action
            if self._invalid_position():
                self._piece_y -= 1
                self._placed = True
            else:
                self._score += 1
        elif action == 4:
            self._piece_x += 1  # Move right
            if self._invalid_position():
                self._piece_x -= 1
        elif action == 5:
            self._piece_y -= 1  # move left
            if self._invalid_position():
                self._piece_y += 1

    def _tick(self):
        # The tetromino falls every frame, unless unable to in which case it sets PLACED to True.
        # The player may take an action per step, so also between ticks
        if self._tick_rate % self.framerate == 1:
            self._tick_rate = 1
            self._piece_y += 1
            if self._invalid_position():
                self._piece_y -= 1
                self._placed = True
        self._tick_rate += 1

    def _clear_lines(self):
        grid = self._get_grid(False)
        idx = np.array([], dtype=int)
        for i in reversed(range(len(grid))):  # Count rows to remove in reverse order
            if grid[i].all():
                idx = np.append(idx, i)
        for i in idx:
            grid = np.delete(grid, i, 0)  # Remove the cleared row
            grid = np.vstack((np.zeros(10), grid))  # Add an empty row on top
            idx += 1  # Since we clear from the bottom any other rows to clear are moved one slot down
        return grid

    def reset(self):
        self._score = 0
        self._current_score = 0
        self._current_lines = 0
        self._current_height = 0
        self._piece_x = 4
        self._piece_y = 1
        self._placed = False
        self._game_over = False
        # Reset pieces
        self._current_piece_n = random.randint(0, 6)
        self._next_piece_n = random.randint(0, 6)
        self._current_piece = shapes[self._current_piece_n][0]
        self._next_piece = shapes[self._next_piece_n][0]
        # Reset board
        self._board = np.zeros([self.height, 10])
        a = np.c_[np.ones(self.height), np.ones(self.height), np.ones(self.height), self._board, np.ones(self.height), np.ones(self.height)]
        self._board = np.vstack((a, np.ones(self.width), np.ones(self.width)))
        self._board[0, self._next_piece_n + 2] = 1
        grid = self._get_grid(False)
        grid = np.vstack((self._board[0][2:12], grid))
        return grid

    def _init_render(self):
        # Rendering Dimensions
        self.screenSize = 600
        self.cell_size = 25
        self.height_r = 20
        self.width_r = 10
        self.top_left_y = self.screenSize / 2 - self.height_r * self.cell_size / 2
        self.top_left_x = self.screenSize / 2 - self.width_r * self.cell_size / 2
        self.screen = pygame.display.set_mode([self.screenSize, self.screenSize])
        # Set up game
        # self.bigfont = pygame.font.Font(None, 80)
        # self.scorefont = pygame.font.Font(None, 30)
        pygame.display.set_caption('Tetris')
        self.background = pygame.Surface(self.screen.get_size())
        self.rendering = True
        self.clock = pygame.time.Clock()

    def render(self, mode="human"):
        if not self.rendering:
            self._init_render()

        # Clear the screen
        self.screen.fill(black)

        # Draw grid
        grid = self._board[3:25, 3:13]  # Pluck out the visual part

        background = (self.top_left_x - 1,
                      self.top_left_y - 1,
                      self.width_r * self.cell_size + 1,
                      self.height_r * self.cell_size + 1)
        pygame.draw.rect(self.screen, grey, background)

        for i in range(self.width_r):
            for j in range(self.height_r):
                val = grid[j, i]
                color = grey if val != 0 else black
                square = (self.top_left_x + int(self.cell_size * i),
                          self.top_left_y + int(self.cell_size * j),
                          self.cell_size - 1, self.cell_size - 1)
                pygame.draw.rect(self.screen, color, square)

        # Draw piece
        size = len(self._current_piece)
        for i in range(size):
            for j in range(size):
                if self._current_piece[i, j] == 0:
                    continue
                square = (self.top_left_x + self.cell_size * (self._piece_x + j - 2),  # POSITION HERE
                          self.top_left_y + self.cell_size * (self._piece_y + i - 2),
                          self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, shape_colors[self._current_piece_n], square)

        pygame.display.flip()
        # Use pygame for rendering
        ...

    @property
    def _board_height(self):
        """Return the height of the board."""
        board = self._get_grid(False)
        # look for any piece in any row
        board = board.any(axis=1)
        # take to sum to determine the height of the board
        return board.sum()

    @property
    def _get_bumpiness(self):
        board = self._get_grid(False)
        # bumpiness is the measure of the difference in heights between neighbouring columns
        bumpiness = 0
        for i in range(9):
            bumpiness += abs(board[:, i].argmax() - board[:, i + 1].argmax())

        return bumpiness

    def _invalid_position(self):
        """
        Returns whether the current position is invalid
        Assumes piece is positioned inside board.
        """
        sub_board = self._board[self._piece_y:self._piece_y+len(self._current_piece), self._piece_x:self._piece_x+len(self._current_piece)]
        # Check for collision by summing and checking for 2
        collision_matrix = self._current_piece + sub_board
        if np.any(collision_matrix == 2):
            return True
        return False

    def rotate(self, clockwise):
        # Check if rotation is valid, then execute accordingly
        pass

    def _get_reward(self):
        reward = 0
        # reward the change in score
        reward += self._score - self._current_score
        # greatly reward a line being cleared
        reward += (self._number_of_lines - self._current_lines) * 100
        # penalize a change in height
        penalty = self._board_height - self._current_height
        # only apply the penalty for an increase in height (not a decrease)
        if penalty > 0:
            # punish the ai for having a bumpy board only when increasing its height
            # until I find a smarter way to calculate bumpiness dependent on placing a piece
            reward -= self._get_bumpiness
            reward -= penalty
        # big penalty for loosing
        if self._game_over:
            reward -= 20
        else:
            reward += 0.01
        # update the locals
        self._current_score = self._score
        self._current_lines = self._number_of_lines
        self._current_height = self._board_height

        return reward
