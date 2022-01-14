# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 08:57:47 2022

@author: Jason
"""

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

headers = ['Name', 'Age', 'Marks']

df = pd.read_csv('Training_13_evening.csv', names=headers)

df.set_index('Name').plot()

plt.show()