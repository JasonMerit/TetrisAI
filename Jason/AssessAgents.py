# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 09:50:23 2022

@author: Jason
"""
import pickle
from Tetris import Tetris
import pandas as pd
import numpy as np

n = 10 # num of games for assessment 
max_gen = 100
header = ["gen", "avg", "var"]
for i in range(n):
    header.append(i+1)

env = Tetris(False)


data = []
for gen in np.arange(10, max_gen+10, 10):
    agent = pickle.load(open('best.pickle_{}'.format(gen), 'rb'))
    print(gen)
    
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
        outputs = [agent.activate(input) for input in evaluations]
    
        # Go to best scored state
        best_index = outputs.index(max(outputs))
        best_state = states[best_index]
        done = env.place_state(best_state) 
        step += 1
        
        if done:
            trial_result.append(step)
            step = 0
            trial += 1
            env.reset()
            done = False
        
        if trial == n:
            noop = np.array(trial_result)
            data.append([gen, np.mean(noop), np.var(noop)] + trial_result)
            quit = True
    
    
    
#print(data)
csv = pd.DataFrame(data, columns=header)
csv.to_csv('Trials.csv', index=False)

    
env.close()


