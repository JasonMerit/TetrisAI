# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 09:50:23 2022

@author: Fabian
"""
from KerasModel import DQN
from Tetris import Tetris
import pandas as pd
import numpy as np


env = Tetris(False, height=20)
agent = DQN(env, state_size=8, epsilon=0)

n = 10  # num of games for assessment
max_gen = 25000
header = ["gen", "avg", "var"]
for i in range(n):
    header.append(i + 1)

env = Tetris(False)

data = []
for game_number in np.arange(0, max_gen + 1000, 5000):
    if game_number > 0:
        agent.load(f'FORFUN323216_{game_number}')
    # Definitions and default settings
    done = False
    quit = False

    trial_result = []
    trial = 0

    while not quit:
        # Let agent determine and take next action
        # Find all final_states and evaluate them
        states = env.get_final_states()
        evaluations = env.get_evaluations(states)

        # Pass the evaluation for each state into the NN
        action, features = agent.take_action(states, evaluations)
        lines_clearead = env.lines_cleared
        done, _ = env.place_state(action)
        # step += 1

        if done:
            trial_result.append(lines_clearead)
            print(f'Model: {game_number}, Trial number: {trial}, Steps: {lines_clearead}')
            trial += 1
            done = False

        if trial == n:
            noop = np.array(trial_result)
            data.append([game_number, np.mean(noop), np.var(noop)] + trial_result)
            quit = True

csv = pd.DataFrame(data, columns=header)
csv.to_csv('Trials_323216.csv', index=False)

env.close()
