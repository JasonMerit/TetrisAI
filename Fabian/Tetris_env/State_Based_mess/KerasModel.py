import tensorflow as tf
import numpy as np
import os
import random
from tensorflow.keras.layers import Dense
from collections import deque

CWD = os.getcwd() + "\\"
WEIGHT_PATH = os.path.join(CWD + 'WEIGHTS')
LOG_DIR = os.path.join(CWD + 'LOGS')


MAX_BUFFER_LENGTH = 100_000
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 512
MINIBATCH_SIZE = 64


class DQN:
    def __init__(self, env, state_size=7, discount=1, epsilon=1, epsilon_min=0.0001, epsilon_decay=0.9995):
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
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='linear'),
        ])

        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mean_squared_error'])

        model.summary()

        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def predict_ratings(self, features):
        """
        Gets an array of features
        returns an array of predictions in the same order
        """
        predictions = [self.model.predict(np.array(features))]
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
        # file_path = os.path.join(WEIGHT_PATH + '\\' + model_name)
        # if Path(file_path).is_file():
        #    self.model.load_weights(file_path)
        # else:
        #    print('No model loaded, file not recognised')
        try:
            self.model.load_weights(model_name)
        except OSError:
            print('Model not found.')

    def save(self, model_name):
        # file_path = os.path.join(WEIGHT_PATH + '\\' + model_name)
        self.model.save_weights(model_name)

    def train(self, games=1000, save=1000, name='DQN'):
        """
        Feed the currently possible actions and their corresponding Features
        to the NN and get its chosen action. Append the transition to memory.
        Do this for a specified number of games, keeping track of the number
        of tetrominos placed (steps)
        returns an array of the scores achieved each game
        """
        scores = []
        steps = 0

        for game in range(1, games + 1):
            print(f'Game: {game} Steps: {steps} AVG Game length: {steps/game}')
            if game % save == 0:
                self.save(f'{name}_{game}')
            actions_list, Features_list, score, done, _ = self.env.reset()
            current_features = np.zeros(len(Features_list[0]), dtype=np.int64)  # set the Features from a new game arbitrarily
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

        for index, (current_features, score, done, future_features) in enumerate(minibatch):
            if not done:
                rating = self.model.predict(future_features.reshape(-1, self.state_size))
                new_q = score + rating[0][0] * self.epsilon_decay
            else:
                new_q = score
            X.append(current_features)
            Y.append(new_q)

        self.model.fit(np.array(X), np.array(Y), batch_size=MINIBATCH_SIZE, verbose=0)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
