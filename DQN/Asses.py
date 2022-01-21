# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 09:50:23 2022

This script plays n number of games with agents extracted at
intervals of 500 games during training and outputs the data to
a csv file.

"""
from KerasModel import DQN
from Tetris import Tetris
import pandas as pd
import numpy as np


env = Tetris(training=True)
agent = DQN(env, state_size=8, epsilon=0)

n = 50  # num of games for assessment
max_game = 36000
header = ["game", "avg", "var"]
for i in range(n):
    header.append(i + 1)

env = Tetris(False)

data = []

for game_number in np.arange(30000, max_game, 500):
    if game_number > 0:
        agent.load(f'Backup_{game_number}')
    # Definitions and default settings
    done = False
    quit = False
    env.reset()
    trial_result = []
    trial = 0

    while not quit:
        # Let agent determine and take next action
        # Find all final_states and evaluate them
        states = env.get_final_states()
        evaluations = env.get_evaluations(states)

        # Pass the evaluation for each state into the NN
        action, features = agent.take_action(states, evaluations)
        lines = env.lines_cleared
        done, _ = env.place_state(action)

        if done:
            trial_result.append(lines)
            trial += 1
            done = False
            env.reset()

        if trial == n:
            noop = np.array(trial_result)
            data.append([game_number, np.mean(noop), np.var(noop)] + trial_result)
            quit = True
            
            csv = pd.DataFrame(data, columns=header)
            csv.to_csv(f'Asses{max_game}.csv', index=False)

env.close()
