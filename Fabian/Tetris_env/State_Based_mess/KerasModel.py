import tensorflow as tf
import numpy as np
import os
import random
from tensorflow.keras.layers import Dense
from collections import deque

CWD = os.getcwd() + "\\"
WEIGHT_PATH = os.path.join(CWD + 'WEIGHTS')
LOG_DIR = os.path.join(CWD + 'LOGS')

MAX_BUFFER_LENGTH = 100000
MIN_REPLAY_MEMORY_SIZE = 512
MINIBATCH_SIZE = 64


class DQN:
    def __init__(self, env, state_size=8, discount=0.99, epsilon=1, epsilon_min=0.0001, epsilon_decay=0.9995):
        self.state_size = state_size
        self.model = self.create_model()
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replay_memory = deque(maxlen=MAX_BUFFER_LENGTH)
        self.env = env

    def create_model(self):
        """Returns a new model."""
        model = tf.keras.models.Sequential([
            Dense(16, input_dim=self.state_size, activation='relu'),
            Dense(1, activation='linear'),
        ])

        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mean_squared_error'])

        model.summary()

        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def take_action(self, actions, Features):
        """
        Takes in array of actions and Features, both arrays of same length
        returns the state chosen by either random action or the NN
        """
        if random.uniform(0, 1) < self.epsilon:
            r = random.randint(0, len(Features) - 1)
            return actions[r], Features[r]

        # pass the features to the model in an array
        ratings = self.model.predict(np.array(Features))
        # get argmax
        max_index = np.argmax(ratings)
        return actions[max_index], Features[max_index]

    def load(self, model_name):
        try:
            self.model.load_weights(model_name)
        except OSError:
            print('Model not found.')

    def save(self, model_name):
        self.model.save_weights(model_name)

    def train(self, games=1000, save=1000, name='DQN'):
        """
        Feed the currently possible actions and their corresponding Features
        to the NN and get its chosen action. Append the transition to memory.
        Do this for a specified number of games, keeping track of the number
        of tetrominos placed (steps)
        returns an array of the scores achieved each game
        """
        steps = 0

        for game in range(1, games + 1):
            if game % 50 == 0:
                print(f'Game: {game} Steps: {steps} AVG {steps/game}')
            if game % save == 0:
                self.save('{}_{}_{}'.format(name, game, steps))
            actions_list, Features_list, score, done = self.env.reset()
            # set the Features from a new game arbitrarily to zero
            current_features = np.zeros(len(Features_list[0]), dtype=np.int64)
            current_score = 0
            while not done:     # Tetris is done when there are no valid actions left
                steps += 1
                action, future_features = self.take_action(actions_list, Features_list)
                actions_list, Features_list, score, done = self.env.step(action)
                self.replay_memory.append((current_features, score, done, future_features))
                current_features = future_features
                current_score += score

            self.learn()

        return steps

    def learn(self):
        """
        When there are enough memories in the buffer this function takes a
        random sample of the memories, rates the future features and updates
        the q-table with these ratings so long as the game was not over
        """
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        X = []
        Y = []

        for i, (current_features, score, done, future_features) in enumerate(minibatch):
            if not done:
                rating = self.model.predict(future_features.reshape(-1, self.state_size))
                new_q = score + rating[0][0] * self.epsilon_decay
            else:
                new_q = 0   # score for game over is 0
            X.append(current_features)
            Y.append(new_q)

        self.model.fit(np.array(X), np.array(Y), batch_size=MINIBATCH_SIZE, verbose=0)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
