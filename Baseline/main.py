import gym, os
from loguru import logger
import numpy as np

from stable_baselines3 import TD3, SAC, PPO, DQN
from sb3_contrib import TRPO

from stable_baselines3.common.noise import NormalActionNoise

from utils import tools, monitor
from loguru import logger

config = {
    "continuous": {
        'env_list': ['HalfCheetah-v2', 'Hopper-v2'],
        'trainers':  ['SAC', 'PPO', 'TD3', 'TRPO']
    },
    "discrete": {
        'env_list': ["LunarLander-v2"],
        'trainers': ['DQN', 'PPO', 'TRPO']
    }
}

def get_model(name, env):
    if  name == 'SAC':
        model = SAC("MlpPolicy", env, verbose=0)
    elif name == 'PPO':
        model = PPO("MlpPolicy", env, verbose=0)
    elif name == 'TD3':
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = TD3('MlpPolicy', env, action_noise=action_noise, verbose=0)
    elif name == 'TRPO':
        model = TRPO("MlpPolicy", env, verbose=0)
    elif name == 'DQN':
        model = DQN("MlpPolicy", env, verbose=0)

    return name.lower(), model

def create_savings(env_id, learner_name):
    log_dir = "model/{}".format(env_id)
    result_dir = "results/{}/{}".format(env_id, learner_name)
    tools.mkdir(log_dir)
    tools.mkdir(result_dir)
    return log_dir, result_dir

if __name__ == '__main__':
    tools.display_torch_device()
    # tools.set_logger_level(1)
    args = tools.load_config('config.yaml')
    mode = args['mode']
    load_model = args['load_model']
    train = args['train']
    max_train_steps = args['max_train_steps']
    reward_decay = args['reward_decay']

    env_list = config[mode]['env_list']
    available_trainers = config[mode]['trainers']

    for i in range(len(env_list)):
        env_id = env_list[i]
        logger.warning('initialize env {}'.format(env_id))
        env = gym.make(env_id)
        env.seed(10)

        for learner_name in available_trainers:
            model = None
            learner_name, model = get_model(name=learner_name, env=env)
            log_dir, result_dir = create_savings(env_id=env_id, learner_name=learner_name)

            logger.warning('using {} model'.format(learner_name))

            best_reward = float('-inf')
            process_monitor = monitor.Process_Monitor(output_dir=result_dir, name=learner_name)
            trainer = tools.Trainer(env, model, process_monitor)

            if load_model:
                filename = os.path.join(log_dir, '{}_best.zip'.format(learner_name))
                trainer.load_model(filename)

            while train and trainer.steps < max_train_steps:
                logger.success('trained in {} using {} {} times - left {} steps'.format(env_id, learner_name, trainer.steps, max_train_steps-trainer.steps))
                reward = trainer.train()

                if reward > best_reward:
                    best_reward = reward
                    filename = os.path.join(log_dir, '{}_best.zip'.format(learner_name))
                    trainer.save_model(filename)
                    logger.success('update best model')

                if trainer.steps % 10 == 0:
                    filename = os.path.join(log_dir, '{}_tmp.zip'.format(learner_name))
                    trainer.save_model(filename)
            
            if train:
                trainer.save_history()

            # tools.save_video(env_id, model, learner_name, result_dir)
            trainer.save_video(video_folder=result_dir, name_prefix=learner_name)
            