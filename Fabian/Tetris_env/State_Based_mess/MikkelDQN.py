from torch.nn import Sequential, Linear, ReLU, MSELoss
from torch.optim import Adam
import numpy as np
from TetEnv_Training import Tetris

no_states = 7
buffer_size = 2000

env = Tetris()

model = Sequential(
    Linear(no_states, 32),
    ReLU(),
    Linear(32, 1)
)

optimizer = Adam(model.parameters(), lr=0.001)
loss = MSELoss()

current_buffer = np.zeros(buffer_size, no_states)
future_buffer = np.zeros(buffer_size, no_states)
reward_buffer = np.zeros(buffer_size, no_states)
done_buffer = np.zeros(buffer_size, 7)

for i in range():
    score = 0
    action, feature = agent.take_action(actions, Features)
    actions, Features, score, done, _ = env.step(action)



