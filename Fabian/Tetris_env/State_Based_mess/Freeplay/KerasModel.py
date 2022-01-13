import tensorflow as tf
import numpy as np
import os
import random
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from collections import deque

WEIGHT_PATH = os.path.dirname(__file__)
LOG_DIR = os.path.dirname(__file__ + 'LOGS')
MAX_BUFFER_LENGTH = 100_000
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 128


class DQN:
    def __init__(self, env, state_size=9, discount=1, epsilon=1, epsilon_min=0.0001, epsilon_decay=9.9995):
        self.state_size = state_size
        self.model = self.create_model()
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replay_memory = deque(maxlen=MAX_BUFFER_LENGTH)
        self.env = env
        self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR,
                                                          histogram_freq=1000,
                                                          write_graph=True,
                                                          write_images=True)

    def create_model(self):
        """Returns a new model."""

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, input_dim=9, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear'),
        ])

        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mean_squared_error'])

        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def predict_ratings(self, features):
        """
        Gets an array of features
        returns an array of predictions in the same order
        """
        predictions = self.model.predict(np.array(features))
        # Each prediction is an array of length 1
        return [predict[0] for predict in predictions]

    def take_action(self, actions, Features):
        """
        Takes in array of [state, features]
        returns the state chosen by either random action or the NN
        """
        if random.uniform(0, 1) < self.epsilon:
            r = random.randint(0, len(Features) - 1)
            random_action_feature = (actions[r], Features[r])
            return random_action_feature

        max_rating = None
        best_action = None
        best_Features = None

        # pass the features to the model in an array
        ratings = self.predict_ratings(Features)

        for index, action in enumerate(actions):
            rating = ratings[index]
            if not max_rating or rating > max_rating:
                max_rating = rating
                best_action = action
                best_Features = Features[index]

        return best_action, best_Features

    def load(self, model_name):
        file_path = os.path.join(WEIGHT_PATH + model_name)
        if Path(file_path).is_file():
            self.model.load_weights(file_path)

    def save(self, model_name):
        file_path = os.path.join(WEIGHT_PATH + model_name)
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        self.model.save_weights(file_path)

    def train(self, games=1000):
        """
        Feed the currently possible actions and their corresponding Features
        to the NN and get its chosen action. Append the transition to memory.
        Do this for a specified number of games, keeping track of the number
        of tetrominos placed (steps)
        returns an array of the scores achieved each game
        """
        scores = []
        steps = 0

        for game in range(games):
            actions_list, Features_list, score, done, _ = self.env.reset()
            current_features = np.zeros(len(Features_list[0]))   # set the Features from a new game arbitrarily
            current_score = 0
            while not done:
                steps += 1
                action, future_features = self.take_action(actions_list, Features_list)
                actions_list, Features_list, score, done, _ = self.env.step(np.array(action))
                if len(actions_list) < 1:
                    break
                self.replay_memory.append((current_features, score, done, future_features))
                current_features = future_features
                current_score += score
            scores.append(score)

            self.learn()

        return scores, steps

    def learn(self):
        # Inspired by DQN-Tetris repo
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        ratings = self.predict_ratings(minibatch)
        X = []
        Y = []

        for index, (current_feature, score, done, future_feature) in enumerate(minibatch):
            if not done:
                new_q = score + ratings[index] * self.epsilon_decay
            else:
                new_q = score
            X.append(current_feature)
            Y.append(new_q)
        self.model.fit(np.array(X), np.array(Y), batch_size=len(X), verbose=0, callbacks=self.tensorboard)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
