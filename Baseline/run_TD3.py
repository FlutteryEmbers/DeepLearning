import gym, os
from loguru import logger
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import matplotlib.pyplot as plt

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common.vec_env import DummyVecEnv
from utils.callbacks import SaveOnBestTrainingRewardCallback
from utils import tools
from loguru import logger
# import mujoco_py
import torch
# mj_path = mujoco_py.utils.discover_mujoco()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
load_model = True
train = True
# tools.set_logger_level(1)
env_id = tools.get_environments(1)
logger.critical('train {} env'.format(env_id))
# Create log dir
log_dir = "tmp/td3/" + env_id
result_dir = "results/td3/{}/".format(env_id)
tools.mkdir(log_dir)
tools.mkdir(result_dir)

# Create and wrap the environment
env = gym.make(env_id)
# env = DummyVecEnv([lambda: gym.make(env_id)])
env = Monitor(env, log_dir)
n_actions = env.action_space.shape[-1]
# Add some action noise for exploration
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Because we use parameter noise, we should use a MlpPolicy with layer normalization
model = TD3('MlpPolicy', env, action_noise=action_noise, verbose=0)

if load_model:
    model = model.load(os.path.join(log_dir, 'best_model.zip'))

if train:
    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    # Train the agent
    timesteps = 3e5

    model.learn(total_timesteps=int(timesteps), callback=callback)

    plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "TD3 LunarLander")
    plt.savefig(result_dir + 'rewards.png')
# tools.render(env, model)
tools.save_video(env_id, model, result_dir)