# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 09:50:23 2022

@author: Jason
"""
import pickle
from Tetris import Tetris
import pandas as pd
import numpy as np
from sys import exit

n = 2 # num of games for assessment 
max_gen = 7
extending = True

header = ["Gen"]

env = Tetris(False)

if not extending:
    data_pieces = []
    data_lines = []
else:
    data_pieces = np.array(pd.read_csv('Trials_pieces.csv'))
    ext_pieces = np.zeros([len(data_pieces), n])
    # three = np.resize(three, len(data_pieces))
    # print(three)
    # data_pieces = np.c_[data_pieces, ext_pieces]
    # data_pieces[0, 4] = 1
    # print(data_pieces)

# exit()


for i, gen in enumerate(np.arange(1, max_gen+1, 1)):
    agent = pickle.load(open('best.pickle_{}'.format(gen), 'rb'))
    
    done = False # True when game over
    quit = False # True when done with current agent
    
    trial_pieces = []
    trial_lines = []
    trial = 0
    step = 0
    
    # Begin trialing n times
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
        done, _ = env.place_state(best_state) 
        step += 1
        
        if done:
            trial_pieces.append(step)
            step = 0
            trial_lines.append(env.lines_cleared)
            env.reset()
            
            trial += 1
            done = False
        
        if trial == n:
            if not extending:
                print(f"{gen}) max_lines: {max(trial_lines)}")
                
                data_pieces.append([gen] + trial_pieces)
                csv = pd.DataFrame(data_pieces)
                csv.to_csv('Trials_pieces.csv', index=False)
                
                data_lines.append([gen] + trial_lines)
                csv = pd.DataFrame(data_lines)
                csv.to_csv('Trials_lines.csv', index=False)
            else:
                print(f"{gen}) max_lines: {max(trial_lines)}")
                
                ext_pieces[i] = np.array(trial_pieces)
                csv = np.c_[data_pieces, ext_pieces]
                csv = pd.DataFrame(csv)
                csv.to_csv('Trials_pieces.csv', index=False)
                
                # data_lines.append([gen] + trial_lines)
                # csv = pd.DataFrame(data_lines)
                # csv.to_csv('Trials_lines.csv', index=False)
                
            
            quit = True
    
    
env.close()


