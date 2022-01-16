# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 08:57:47 2022

@author: Jason
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Trials_14.csv')

data = np.array(df)


X = np.arange(1, len(data)+1) * 10
# X = np.linspace(10, len(data)*10+1)
Y = data[:, 1]


ci = 2/np.sqrt(len(data[0]) - 3)

fig, ax = plt.subplots()
ax.plot(X, Y)
ax.fill_between(X, (1-ci)*Y, (1+ci)*Y, color='b', alpha=.1)


plt.xlabel("Generation")
plt.ylabel("avg_lines_cleared")
plt.show