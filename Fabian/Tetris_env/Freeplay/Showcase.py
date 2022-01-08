from stable_baselines3.common.callbacks import BaseCallback
import os
from TetEnv import Tetris
import gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_checker import check_env
import numpy as np
import time

env = Tetris()
env = DummyVecEnv([lambda:env])
state = env.reset()
CHECKPOINT_DIR = 'train/'

model = DQN.load(CHECKPOINT_DIR + 'best_DQN_model_2600052.zip')

while True:
    action = model.predict(state)
    state, reward, done, score = env.step(action)
    if reward:
        print(reward)
    env.render()
    if done:
        env.reset()
