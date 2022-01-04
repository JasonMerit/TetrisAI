import numpy as np

A = np.ones((5,5))
B = np.array([[ 0.1,  0.2],
       [ 0.3,  0.4]])

A[3:5, 3:5] = B


a, b = 0, 1 if True else 1, 2

