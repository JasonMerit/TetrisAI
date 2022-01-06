import tensorflow as tf
import numpy as np
import os
import random
from tensorflow import keras
from tensorflow.keras import layers
from TetEnv import Tetris
from pathlib import Path

WEIGHT_PATH = os.path.dirname(__file__)


class ExperienceBuffer:
    def __init__(self, buffer_size=20000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return random.sample(self.buffer, size)


class DQN:
    def __init__(self):
        # Configuration paramaters for the whole setup
        self.gamma = 0.99  # Discount factor for past rewards
        self.epsilon = 1.0  # Epsilon greedy parameter
        self.epsilon_min = 0.001  # Minimum epsilon greedy parameter
        self.epsilon_max = 1.0  # Maximum epsilon greedy parameter
        self.experiences = ExperienceBuffer()
        self.model = self.create_model()

    def create_model(self):
        """Returns a new model."""

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, input_dim=9, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(7, activation='linear'),
        ])

        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mean_squared_error'])

        return model

    def prediction(self, state):
        return self.model.predict(state)

    def load(self, model_name):
        file_path = os.path.join(WEIGHT_PATH + model_name)
        if Path(file_path).is_file():
            self.model.load_weights(file_path)

    def save(self, model_name):
        file_path = os.path.join(WEIGHT_PATH + model_name)
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        self.model.save_weights(file_path)

    def learn(self, batch_size=512, epochs=1):
        if len(self.experiences.buffer) < batch_size:
            return

        batch = self.experiences.sample(batch_size)
        x = []
        y = []

        for (state, reward, done, _) in enumerate(batch):
            if not done:
                q = reward + 1


model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, input_dim=7, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(7, activation='linear'),
        ])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mean_squared_error'])

a = np.ones([0,2,7])
a.reshape(-1, 7)

print(model.predict(a))
