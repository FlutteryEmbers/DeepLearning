from locale import normalize
from traceback import StackSummary
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise


import gym
import os
import numpy as np
from utils import tools, monitor
import torch.nn as nn

env_id = "Hopper-v3"
env = gym.make(env_id)
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
model_name = "TD3"
model = TD3('MlpPolicy', env, action_noise=action_noise, verbose=0, device='cuda:0', train_freq=1, learning_rate=3*(10**-4), gradient_steps=1, batch_size=256)
'''
net_arch =  {"pi":[256, 256], "vf":[256,256]}
policy_kwargs = {
    "log_std_init": -2,
    "ortho_init": False,
    "activation_fn": nn.ReLU,
    "net_arch": net_arch
}

policy_kwargs = dict(log_std_init=-2, ortho_init=False, activation_fn=nn.ReLU)
model_name = "PPO"
model = PPO("MlpPolicy", env, verbose=0, device='cuda:0', 
            normalize_advantage=True, batch_size=32, n_steps=512, gamma=0.999, learning_rate=9.80828*(10**-6), ent_coef=0.00229519,
            clip_range=0.2, n_epochs=5, gae_lambda=0.99, max_grad_norm=0.7, vf_coef=0.835671, policy_kwargs=policy_kwargs)
'''

max_train_steps = 1000000

best_reward = float('-inf')
process_monitor = monitor.Process_Monitor(output_dir="assets/experts", name=model_name)

trainer = tools.Trainer(env, model, process_monitor)
log_dir = "assets/experts"
while trainer.steps < max_train_steps:
    print('{}: {} times - left {} steps'.format(env_id, trainer.steps, max_train_steps-trainer.steps))
    reward = trainer.train()

    if reward > best_reward:
        best_reward = reward
        filename = os.path.join(log_dir, '{}_{}_best.zip'.format(model_name, env_id))
        trainer.save_model(filename)
        print('update best model')

    if trainer.steps % 1000 == 0:
        filename = os.path.join(log_dir, '{}_{}_tmp.zip'.format(model_name, env_id))
        trainer.save_model(filename)

trainer.save_history()

# tools.save_video(env_id, model, learner_name, result_dir)
trainer.save_video(video_folder=log_dir, name_prefix='{}_{}'.format(model_name, env_id))
