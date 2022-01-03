from TetrisAI import Tetris
import numpy as np
import pygame

# Initialize the environment
env = Tetris()
env.reset()
x, y, board = env.get_state()