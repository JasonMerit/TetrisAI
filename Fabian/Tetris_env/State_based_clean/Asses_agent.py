# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 09:50:23 2022

@author: Fabian
"""
from KerasModel import DQN
from Tetris import Tetris

fps = 15

env = Tetris(training=False)

agent = DQN(env, state_size=5, epsilon=0)
agent.load('32x2_24000_1149683')

# Definitions and default settings
run = True
action_taken = False
slow = True
done = False
games = 50
pieces_placed = 0

for i in range(games):
    env.reset()
    done = False
    while not done:
        pieces_placed += 1
        # Let agent determine and take next action
        # Find all final_states and evaluate them
        states = env.get_final_states()
        evaluations = env.get_evaluations(states)

        # Pass the evaluation for each state into the NN
        action, features = agent.take_action(states, evaluations)
        # Go to best scored state
        done = env.place_state(action)
    

