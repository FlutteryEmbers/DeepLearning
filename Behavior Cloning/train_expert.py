from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gym
import os
import numpy as np
from utils import tools, monitor
# env = make_vec_env("HalfCheetah-v2", n_envs=2)
env = gym.make("HalfCheetah-v2")
model = PPO("MlpPolicy", env, verbose=0)

max_train_steps = 100
best_reward = float('-inf')
process_monitor = monitor.Process_Monitor(output_dir="assets/experts", name='ppo')
trainer = tools.Trainer(env, model, process_monitor)
log_dir = "assets/experts"
while trainer.steps < max_train_steps:
    print('HalfCheetah-v2: {} times - left {} steps'.format(trainer.steps, max_train_steps-trainer.steps))
    reward = trainer.train()

    if reward > best_reward:
        best_reward = reward
        filename = os.path.join(log_dir, 'ppo_best.zip')
        trainer.save_model(filename)
        print('update best model')

    if trainer.steps % 10 == 0:
        filename = os.path.join(log_dir, 'ppo_tmp.zip')
        trainer.save_model(filename)

trainer.save_history()

# tools.save_video(env_id, model, learner_name, result_dir)
trainer.save_video(video_folder=log_dir, name_prefix='ppo')
