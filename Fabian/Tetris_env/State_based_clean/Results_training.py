# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 08:57:47 2022

@author: Jason
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(f'CSVDATA1')

data = np.array(df)


X = data[:, 0]
Y = data[:, 2]


ci = 0.1
fig, ax = plt.subplots()
ax.plot(X, Y)
ax.fill_between(X, (1-ci)*Y, (1+ci)*Y, color='b', alpha=.1)

plt.title(f'Lines vs games')
plt.xlabel("Games played")
plt.ylabel("Total lines cleared")
plt.show()

Y = [0]

for i in range(1, len(data[:, 0])):
    Y.append(data[i, 0] - data[i - 1, 0])

X = data[:, 3]


ci = 2/np.sqrt(len(data[0]) - 3)
# ci = 0.2
fig, ax = plt.subplots()
ax.plot(X, Y)
# ax.fill_between(X, (1-ci)*Y, (1+ci)*Y, color='b', alpha=.1)

plt.title(f'Games vs time')
plt.ylabel("Games played")
plt.xlabel("Time")
plt.show()


