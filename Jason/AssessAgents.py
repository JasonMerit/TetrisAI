# Assessing agents

import pickle
from Tetris import Tetris
import pandas as pd
import numpy as np


n = 50 # num of games for assessment 
extending = False # True if adding to already existing data
max_gen = 150
file_name = "Trials_17_eve.csv"
# REMEBER TO CHANGE AGENT LOAD

# Weights for BCTS
#W = np.array([-12.63, 6.6, -9.22, -19.77, -13.08, -10.49, -1.61, -24.04])

env = Tetris(False)

if not extending:
    data = []
else:
    data = np.array(pd.read_csv(file_name))
    ext_lines = np.zeros([len(data), n])


for i, gen in enumerate(np.arange(10, max_gen+10, 10)):
    agent = pickle.load(open('best.pickle_{}'.format(gen), 'rb'))
    
    done = False # True when game over
    quit = False # True when done with current agent
    
    trial_lines = []
    trial = 0
    
    # Begin trialing n times
    while not quit:    
        # Let agent determine and take next action
        # Find all final_states and evaluate them
        states = env.get_final_states()
        evaluations = env.get_evaluations(states)
        
        # Pass the evaluation for each state into the NN
        outputs = [agent.activate(input) for input in evaluations]
        # outputs = [np.dot(input, W) for input in evaluations]
    
        # Go to best scored state
        best_index = outputs.index(max(outputs))
        best_state = states[best_index]
        done, _ = env.place_state(best_state)
        
        if done:
            trial_lines.append(env.lines_cleared)
            env.reset()
            
            trial += 1
            done = False
        
        if trial == n:
            print(f"{gen}) max_lines: {max(trial_lines)}")
            if not extending:
                data.append([gen] + trial_lines)
                csv = pd.DataFrame(data)
                csv.to_csv(file_name, index=False)
            else:
                ext_lines[i] = np.array(trial_lines)
                csv = np.c_[data, ext_lines]
                csv = pd.DataFrame(csv)
                csv.to_csv(file_name , index=False)
            
            quit = True
    
env.close()


