# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 08:57:47 2022

@author: Jason
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dqn_trials = 'Trialsbackup16'
dqn_training = 'CSVDATAbackup'
neat_trials = 'Trials_17_h16'
neat_training = 'Training_17_eve'
linear_lines = 'Trials_Linear16'

data = []
for i in range(1, 7):
    df = pd.read_csv(f'{dqn_trials}{i*6000}.csv')
    tem = np.array(df)
    data.append([np.mean(row) for row in tem[:, 3:]])
df2 = pd.read_csv(f'{dqn_training}')
data2 = np.array(df2)
df3 = pd.read_csv(f'{neat_trials}')
data3 = np.array(df3)
df4 = pd.read_csv(f'{neat_training}.csv')
data4 = np.array(df4)
df5 = pd.read_csv(f'{linear_lines}')
data5 = np.array(df5)

X = np.linspace(0, 36, 72)
Y1 = np.array([n for m in data for n in m])
Y2 = data2[:72, 3]
Y3 = np.array([np.mean(row) for row in data3[0:72, 1:]])
Y4 = data4[:72, 3]
Y5 = np.array([np.mean(data5) for i in range(72)])

print(np.argmax(Y1) * 500, Y1)
print(len(X), len(Y1), len(Y2), len(Y3), len(Y4))
ci = 2/np.sqrt(len(Y1) - 1)
ci2 = 2/np.sqrt(len(data3[0]) - 1)
ci3 = 2/np.sqrt(len(data5))
fig, ax1 = plt.subplots()

color1 = 'tab:red'
color2 = 'tab:green'
color3 = 'tab:cyan'
ax1.set_xlabel('Games played (1000s)')
ax1.set_ylabel('Average lines cleared')
ax1.plot(X, Y1, color=color1)
ax1.plot(X, Y3, color=color2)
ax1.plot(X, Y5, color=color3)
ax1.fill_between(X, (1-ci) * Y1, (1 + ci) * Y1, color=color1, alpha=.1)
ax1.fill_between(X, (1-ci2) * Y3, (1 + ci2) * Y3, color=color2, alpha=.1)
ax1.fill_between(X, (1-ci3) * Y5, (1 + ci3) * Y5, color=color3, alpha=.1)
ax1.tick_params(axis='y')
# ax1.legend(['DQN lines cleared', 'NEAT lines cleared'], 'upper_left')

ax2 = ax1.twinx()

ax2.set_ylabel('Training time')
ax2.plot(X, Y2, ':', color=color1)
ax2.plot(X, Y4, ':', color=color2)
ax2.tick_params(axis='y')

ax2.legend(['DQN training time', 'NEAT training time'])

fig.tight_layout()
plt.show()
