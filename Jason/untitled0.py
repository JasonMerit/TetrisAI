# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 11:38:30 2022

@author: Jason
"""

import numpy as np

board = np.zeros([22,10])
wall = np.ones([22, 2])
floor = np.ones([2, 14])
board = np.c_[wall, board, wall]
board = np.vstack((board, floor))

print(board)