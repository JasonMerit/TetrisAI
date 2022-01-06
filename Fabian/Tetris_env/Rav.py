import numpy as np
from TetEnv_States import Tetris

k = [1,1,1]
l, m, n = k
print(l,m,n)
env = Tetris()
env.render()
env.step([5,5,0])
env.render()