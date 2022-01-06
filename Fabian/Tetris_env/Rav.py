import numpy as np


sum = 0
board = np.ones([20, 10])
board[:, 1] = 0
for i in range(1, 10):
    sum += abs(board[:, i].sum() - board[:, i - 1].sum())

print(sum)
