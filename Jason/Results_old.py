# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 08:57:47 2022

@author: Jason
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Trials_17.csv')
r_df = pd.read_csv('Trials_15.csv')

data = np.array(df)
rdata = np.array(r_df)


X = np.arange(1, len(data)+1) * 10
rX = np.arange(1, len(rdata)+1) * 10
# X = np.linspace(10, len(data)*10+1)
Y = data[:, 1]
rY = rdata[:, 1]


ci = 2/np.sqrt(len(data[0]) - 2) # Game, avg

fig, ax = plt.subplots()
ax.plot(X, Y, color = 'b')
ax.plot(rX, rY, color = 'r')
ax.fill_between(X, (1-ci)*Y, (1+ci)*Y, color='b', alpha=.1)
ax.fill_between(rX, (1-ci)*rY, (1+ci)*rY, color='r', alpha=.1)


plt.xlabel("Generation")
plt.ylabel("avg_lines_cleared")
plt.show