import pandas as pd
import numpy as np

df = pd.read_csv('Best.csv')
data = np.array(df)
for i in range(2, 5):
    df = pd.read_csv(f'Best{i}.csv')
    data = np.append(data, np.array(df))

df = pd.read_csv('Trials_Linear_16h.csv')
data2 = np.array(df)
for i in range(1, 4):
    df = pd.read_csv(f'Trials_Linear_v{i}_16.csv')
    data2 = np.append(data2, np.array(df))

df = pd.read_csv(f'Trials_230.csv')
data3 = np.array([])
for i in range(1, 3):
    df = pd.read_csv(f'Trials_230_v{i}.csv')
    data3 = np.append(data3, np.array(df))

print(len(data))
print(len(data2))
print(len(data3))
ci = 2/np.sqrt(len(data))
mean = np.mean(data)
ci2 = 2/np.sqrt(len(data2))
mean2 = np.mean(data2)
ci3 = 2/np.sqrt(len(data3))
mean3 = np.mean(data3)

print('DQN:')
print(f'Highest score: {np.max(data)}, Average score: {mean}, Confidence bounds: {ci * 100} %')
print(f'Upper bound: {mean*ci + mean} \nLower bound: {-mean*ci+mean}')
print(f'Plus minus {ci*mean}')
print('NEAT:')
print(f'Highest score: {np.max(data3)}, Average score: {mean3}, Confidence bounds: {ci3 * 100} %')
print(f'Upper bound: {mean3*ci3 + mean3} \nLower bound: {-mean3*ci3+mean3}')
print(f'Plus minus {ci3*mean3}')
print('Linear model:')
print(f'Highest score: {np.max(data2)}, Average score: {mean2}, Confidence bounds: {ci2 * 100} %')
print(f'Upper bound: {mean2*ci2 + mean2} \nLower bound: {-mean2*ci2+mean2}')
print(f'Plus minus {ci2*mean2}')
