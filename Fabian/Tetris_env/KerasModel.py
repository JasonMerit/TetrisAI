import tensorflow as tf
import numpy as np
import os
import random
from tensorflow import keras
from tensorflow.keras import layers
from TetEnv import Tetris


class DQN:
    def __init__(self, state_size, ):
        # Configuration paramaters for the whole setup
        self.state_size = state_size
        self.gamma = 0.99  # Discount factor for past rewards
        self.epsilon = 1.0  # Epsilon greedy parameter
        self.epsilon_min = 0.1  # Minimum epsilon greedy parameter
        self.epsilon_max = 1.0  # Maximum epsilon greedy parameter
        batch_size = 32  # Size of batch taken from replay buffer
        max_steps_per_episode = 10000

        env = Tetris()

    def create_model(self):
        """Returns a new model."""

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear'),
        ])

        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mean_squared_error'])

        return model


