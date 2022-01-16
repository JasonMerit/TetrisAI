# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 09:50:23 2022

@author: Fabian
"""
from KerasModel import DQN
from Tetris import Tetris
import pandas as pd
import numpy as np


env = Tetris(training=False)
agent = DQN(env, state_size=5, epsilon=0)

n = 50  # num of games for assessment
max_gen = 10_000
header = ["gen", "avg", "var"]
for i in range(n):
    header.append(i + 1)

env = Tetris(False)

data = []
for game_number in np.arange(1000, max_gen + 1000, 1000):
    agent.load(f'DQN_{game_number}')
    # Definitions and default settings
    done = False
    quit = False

    trial_result = []
    trial = 0
    step = 0

    while not quit:
        # Let agent determine and take next action
        # Find all final_states and evaluate them
        states = env.get_final_states()
        evaluations = env.get_evaluations(states)

        # Pass the evaluation for each state into the NN
        action, features = agent.take_action(states, evaluations)
        done = env.place_state(action)
        step += 1

        if done:
            trial_result.append(step)
            step = 0
            trial += 1
            env.reset()
            done = False

        if trial == n:
            noop = np.array(trial_result)
            data.append([game_number, np.mean(noop), np.var(noop)] + trial_result)
            quit = True

csv = pd.DataFrame(data, columns=header)
csv.to_csv('Trials.csv', index=False)

env.close()
