import pygame
import numpy as np
import random
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

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


def new_board():
    board = np.zeros([22, 10])
    wall = np.ones([22, 2])
    floor = np.ones([2, 15])
    board = np.c_[np.ones(22), wall, board, wall]
    board = np.vstack((board, floor))
    return board


class Tetris(gym.Env):
    metadata = {'render.modes': ['human']}
    # Rendering?
    rendering = False

    # Rendering Dimensions
    screen_size = 600
    cell_size = 25
    height = 20
    width = 10
    top_left_y = screen_size / 2 - height * cell_size / 2
    top_left_x = screen_size / 2 - width * cell_size / 2
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

    def __init__(self, movement):
        pygame.init()
        self.action_space = spaces.Discrete(len(movement))
        # Observation space contains the board, and an extra row representing the next piece
        self.observation_space = spaces.Box(low=0, high=1, shape=(210, 1), dtype=int)
        self.current_score = 0
        self.score = 0
        self.current_lines = 0
        self.current_height = 0
        self.number_of_lines = 0
        self.board = new_board()
        self.piece = Piece()
        self.next_piece = Piece()

    def step(self, action):
        # Move piece and undo if invalid move
        if action == 1:
            self.piece.x -= 1
            if not self._valid_position():
                self.piece.x += 1
        elif action == 2:
            self.piece.x += 1
            if not self._valid_position():
                self.piece.x -= 1
        elif action == 3:
            self.piece.y += 1
            if not self._valid_position():
                self.piece.y -= 1
                self.new_piece()
        elif action == 4:
            self.piece.rotate()
            if not self._valid_position():
                self.piece.rotate(False)
        elif action == 5:
            self.piece.rotate(False)
            if not self._valid_position():
                self.piece.rotate(True)
        elif action == 6:
            while self._valid_position():
                self.piece.y += 1
            self.piece.y -= 1
            self.new_piece()

        reward = self.get_reward()

        return self.get_state(), reward, self.game_over(), {}

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
        if self.game_over():
            reward -= 20
        else:
            reward += 0.01
        # update the locals
        self._current_score = self.score
        self._current_lines = self.number_of_lines
        self._current_height = self.board_height()

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
        # Let piece drop
        self.piece.y += 1
        if not self._valid_position():
            self.piece.y -= 1
            self.new_piece()

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
        if not self.rendering:
            self.init_render()

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
        self.board = new_board()
        self.piece = Piece()

        return self.get_state()

    def close(self):
        pygame.quit()

    def init_render(self):
        self.screen = pygame.display.set_mode([self.screen_size, self.screen_size])
        pygame.display.set_caption('Tetris')
        self.background = pygame.Surface(self.screen.get_size())
        self.rendering = True
        self.clock = pygame.time.Clock()

        # Set up game
        self.bigfont = pygame.font.Font(None, 80)
        self.scorefont = pygame.font.Font(None, 30)

    def get_state(self):
        next_piece_position = np.zeros(10)
        next_piece_position[self.next_piece.tetromino] = 1
        observation = np.concatenate((self.board[2:22, 3:13].flat, next_piece_position.flat))
        return observation.reshape(210, 1)


# Initialize the environment
movement_list = [0, 1, 2, 3, 4, 5, 6]
env = Tetris(movement_list)
env.reset()

# Definitions and default settings
run = False
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
                action, action_taken = 1, True
            if event.key == pygame.K_LEFT:
                action, action_taken = 2, True
            if event.key == pygame.K_UP:
                action, action_taken = 3, True
            if event.key == pygame.K_DOWN:
                action, action_taken = 4, True
            if event.key == pygame.K_z:
                action, action_taken = 5, True
            elif event.key == pygame.K_x:
                action, action_taken = 6, True
            elif event.key == pygame.K_SPACE:
                action, action_taken = 7, True
            elif event.key == pygame.K_e:
                action, action_taken = 8, True

    # AI controller
    if runai:
        pass

    # Human controller
    else:
        if action_taken:
            env.step(action)
            action_taken = False
            print(env.get_state())

    if render:
        env.render()

env.close()
