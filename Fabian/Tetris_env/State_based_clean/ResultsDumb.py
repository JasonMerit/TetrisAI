# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 08:57:47 2022

@author: Jason
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dqn_trials = 'Trialsprim'
dqn_training = 'CSVDATA1'
neat_trials = 'Trials_17_eve'
neat_training = 'Training_17_eve'

df = pd.read_csv(f'{dqn_trials}.csv')
data = np.array(df)
df2 = pd.read_csv(f'{dqn_training}')
data2 = np.array(df2)
df3 = pd.read_csv(f'{neat_trials}.csv')
data3 = np.array(df3)
df4 = pd.read_csv(f'{neat_training}.csv')
data4 = np.array(df4)

X = data[1:, 0]
Y1 = np.array([np.mean(row) for row in data[1:, 3:]])
Y2 = data2[:, 3]
Y3 = np.array([np.mean(row) for row in data3[0:71, 1:]])
Y4 = data4[0:71, 3]

print(len(X), len(Y1), len(Y2), len(Y3), len(Y4))
print(Y3[0], Y4[0])
ci = 2/np.sqrt(len(data[0]) - 3)
fig, ax1 = plt.subplots()

color1 = 'tab:red'
color2 = 'tab:blue'
ax1.set_xlabel('Games played')
ax1.set_ylabel('Average lines cleared')
ax1.plot(X, Y1, color=color1)
ax1.plot(X, Y3, color=color2)
ax1.fill_between(X, (1-ci) * Y1, (1 + ci) * Y1, color='r', alpha=.1)
ax1.fill_between(X, (1-ci) * Y3, (1 + ci) * Y3, color='b', alpha=.1)
ax1.tick_params(axis='y')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('Training time')  # we already handled the x-label with ax1
ax2.plot(X, Y2, color=color1)
ax2.plot(X, Y4, color=color2)
ax2.tick_params(axis='y')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
