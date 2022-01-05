from stable_baselines3.common.callbacks import BaseCallback
import os
from TetEnv import Tetris
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_checker import check_env
import numpy as np

class TrainLogging(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainLogging, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f'best_PPO_model_{self.n_calls}')
            self.model.save(model_path)

        return True


CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

callback = TrainLogging(check_freq=50000, save_path=CHECKPOINT_DIR)
env = Tetris()
check_env(env)
env = DummyVecEnv([lambda:env])

model = PPO(policy='MlpPolicy', env=env, tensorboard_log=LOG_DIR)

model.learn(total_timesteps=2000000, callback=callback)
