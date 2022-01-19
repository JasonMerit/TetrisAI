# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 14:38:53 2022

@author: Jason
"""

import pickle
# from Tetris import Tetris
# import pygame
# import numpy as np

agent = pickle.load(open('best.pickle', 'rb'))
print(len(agent.input_nodes))
print(len(agent.node_evals))
print(len(agent.output_nodes))
print(agent.values)
