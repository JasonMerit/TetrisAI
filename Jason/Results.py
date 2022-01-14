# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 08:57:47 2022

@author: Jason
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Trials_13_eve.csv')

data = np.array(df)

X = np.arange(1, len(data)) * 10
print(len(X))
Y = data[1:, 1]

plt.plot(X, Y)
plt.xlabel("Generation")
plt.ylabel("avg_pieces_placed")
plt.show