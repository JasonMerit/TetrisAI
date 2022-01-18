# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 08:57:47 2022

@author: Jason
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(f'CSVDATA')

data = np.array(df)


X = np.arange(1, len(data)+1) * 5000
# X = np.linspace(10, len(data)*10+1)
Y = data[:, 1]


ci = 2/np.sqrt(len(data[0]) - 3)
# ci = 0.2
fig, ax = plt.subplots()
ax.plot(X, Y)
ax.fill_between(X, (1-ci)*Y, (1+ci)*Y, color='b', alpha=.1)

plt.title(f'Model layers: {model}')
plt.xlabel("Games played")
plt.ylabel("Average lines cleared")
plt.show()
