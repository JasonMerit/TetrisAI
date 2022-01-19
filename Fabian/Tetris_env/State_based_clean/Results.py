# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 08:57:47 2022

@author: Jason
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_name = 'Trials_REAL'
training_file = 'CSVDATA1'

df = pd.read_csv(f'{file_name}.csv')
data = np.array(df)
df2 = pd.read_csv(f'{training_file}')
data2 = np.array(df2)


X = data[1:, 0]
Y1 = []
for row in data[1:, 3:]:
    Y1.append(np.mean(row))
Y1 = np.array(Y1)
Y2 = data2[:, 3]
print(len(X), len(Y1), len(Y2))

ci = 2/np.sqrt(len(data[0]) - 3)
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Games played')
ax1.set_ylabel('Average lines cleared', color=color)
ax1.plot(X, Y1, color=color)
ax1.fill_between(X, (1-ci) * Y1, (1 + ci) * Y1, color='b', alpha=.1)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Training time', color=color)  # we already handled the x-label with ax1
ax2.plot(X, Y2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
