import os
import numpy as np
from utils import tools
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from loguru import logger
import matplotlib.pyplot as plt
import pickle

class Process_Monitor():
	def __init__(self, output_dir='', name='') -> None:
		self.rewards = []
		self.name = name
		self.output_dir = output_dir + '/'
	
	def store(self, reward):
		self.rewards.append(reward)
		logger.info('reward is {}'.format(reward))

	def average(self, n):
		reward = np.array(self.rewards)
		mean = np.mean(reward)
		logger.info('average rewards of last {} evaluation is {}'.format(n, mean))
		return mean

	def plot_learning_curve(self):
		name = '{}_rewards'.format(self.name)
		filename = self.output_dir + name + '.png'

		x = np.arange(0, len(self.rewards))

		plt.plot(x, self.rewards)
		plt.title(label=name)
		plt.savefig(filename)
		logger.success('successfully create {}'.format(filename))
		plt.close()

	def plot_average_learning_curve(self, n):
		name = '{}_average_{}_rewards'.format(self.name, n)
		filename = self.output_dir + name + '.png'

		reward = np.array(self.rewards)
		means = np.zeros(len(self.rewards))
		x = np.arange(len(self.rewards))
		for i in range(len(means)):
			means[i] = np.mean(reward[max(0, i-n):(i+1)])

		plt.plot(x, means)
		plt.title(label=name)
		plt.savefig(filename)
		logger.success('successfully create {}'.format(filename))
		plt.close()
	
	def dump_to_file(self):
		filename = self.output_dir + self.name + '_'
		with open(filename + 'history_rewards.pickle', 'wb') as handle:
			pickle.dump(self.rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)

		handle.close()

	def load_from_file(self, file):
		with open(file, 'rb') as handle:
			self.rewards = pickle.load(handle)
		handle.close()

	def reset(self):
		self.rewards = []

class SaveOnBestTrainingRewardCallback(BaseCallback):
	"""
	Callback for saving a model (the check is done every ``check_freq`` steps)
	based on the training reward (in practice, we recommend using ``EvalCallback``).

	:param check_freq:
	:param log_dir: Path to the folder where the model will be saved.
	It must contains the file created by the ``Monitor`` wrapper.
	:param verbose: Verbosity level.
	"""
	def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
		super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
		self.check_freq = check_freq
		self.log_dir = log_dir
		self.save_path = os.path.join(log_dir, 'best_model')
		self.best_mean_reward = -np.inf

	def _init_callback(self) -> None:
		# Create folder if needed
		tools.mkdir(self.save_path)

	def _on_step(self) -> bool:
		if self.n_calls % self.check_freq == 0:
			x, y = ts2xy(load_results(self.log_dir), 'timesteps')
			if len(x) > 0:
			# Mean training reward over the last 100 episodes
				mean_reward = np.mean(y[-100:])
			if self.verbose > 0:
				logger.info(f"Num timesteps: {self.num_timesteps}")
				logger.info(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

			# New best model, you could save the agent here
			if mean_reward > self.best_mean_reward:
				self.best_mean_reward = mean_reward
				# Example for saving best model
				if self.verbose > 0:
					logger.success(f"Saving new best model to {self.save_path}")
				self.model.save(self.save_path)

		return True