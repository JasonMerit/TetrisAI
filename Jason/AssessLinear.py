# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 09:50:23 2022

@author: Jason
"""

from Tetris import Tetris
import numpy as np
import pandas as pd

W = np.array([-12.63, 6.6, -9.22, -19.77, -13.08, -10.49, -1.61, -24.04])

env = Tetris(True)

# Definitions and default settings
N = 10000 # num of games for assessment 
file_name = "Trials_Linear.csv"

data = []

for n in np.arange(1, N+1):
    done = False
    while not done:
        # Let agent determine and take next action
        # Find all final_states and evaluate them
        states = env.get_final_states()
        evaluations = env.get_evaluations(states)
        
        # Pass the evaluation for each state into the NN
        # outputs = [agent.activate(input) for input in evaluations]
        outputs = [np.dot(input, W) for input in evaluations]
        
    
        # Go to best scored state
        best_index = outputs.index(max(outputs))
        best_state = states[best_index]
        done, _ = env.place_state(best_state) 
        
        if done:
            lc = env.lines_cleared
            data.append(lc)
            print(f"{n}) lines: {lc}")
            env.reset()
            csv = pd.DataFrame(data)
            csv.to_csv(file_name, index=False)
        
        # if n % 100 == 0: # Export every 100 games
        
    
    
    
    




