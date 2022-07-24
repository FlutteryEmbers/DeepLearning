from distutils.log import info
import os
import numpy as np
from utils import tools
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from loguru import logger

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