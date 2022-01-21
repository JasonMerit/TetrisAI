# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 09:50:23 2022

@author: Fabian
"""
from KerasModel import DQN
from Tetris import Tetris
import pandas as pd
import numpy as np


env = Tetris(training=True)
agent = DQN(env, state_size=8, epsilon=0)

n = 30  # num of games for assessment
max_gen = 36000
header = ["gen", "avg", "var"]
for i in range(n):
    header.append(i + 1)

env = Tetris(False)

data = []

for game_number in np.arange(30000, max_gen, 500):
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
        lines= env.lines_cleared
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
            csv.to_csv(f'Trialsbackup16{max_gen}.csv', index=False)

env.close()
