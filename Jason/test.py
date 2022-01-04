# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 13:18:03 2022

@author: Jason
"""
import numpy as np
S = np.array([[[0,0,0],
               [0,1,1],
               [1,1,0]],
              [[0,1,0],
               [0,1,1],
               [0,0,1]]])
x, y = 9, 10

board = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], #[9, 10]
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

def get_sub_board():
    size = len(S[0])
    sx = max(size + x - 13, 0) # Determine if sub-board exceeds board
    sy = max(size + y - 23, 0) # 0 if it doesn't
    print(sx, sy)
    n1, n2 = np.arange(size - sx) + x, np.arange(size-sy) + y
    sub_board = board[n2[:,None], n1[None,:]]
    return sub_board

a = get_sub_board()
print(a)


# Jeg vil gerne kunne omskrive board til en tuppel af koordinator
if False:
    A = np.array([[1, 1, 0], [1, 0, 1], [0, 0, 1]])
    print(A)
    
    indices = np.where(A == 0)
    a, b = indices[0],indices[1]
    coor = np.array(list(zip(a,b)))
    print(coor)

if False:
    x, y = 2, 3
    size = len(S[0])
    for i in np.arange(size) + x:
        for j in np.arange(size) + y:
            print(i,j)
    