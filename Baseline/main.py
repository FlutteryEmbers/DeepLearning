import gym, os
from loguru import logger
import numpy as np

from stable_baselines3 import TD3, SAC, PPO
from sb3_contrib import TRPO

from stable_baselines3.common.noise import NormalActionNoise

from utils import tools, monitor
from loguru import logger
# import mujoco_py
# mj_path = mujoco_py.utils.discover_mujoco()

learners = ['SAC', 'PPO', 'TD3', 'TRPO']
env_list = ["LunarLanderContinuous-v2", 'HalfCheetah-v2', 'Hopper-v2']
# reward_threshold = []

def get_environments(choice):
    logger.critical('Trainning: {}'.format(env_list[choice]))
    return env_list[choice]

def get_model(index, env):
    if  index == 0:
        model = SAC("MlpPolicy", env, verbose=0)
    elif index == 1:
        model = PPO("MlpPolicy", env, verbose=0)
    elif index == 2:
        n_actions = env.action_space.shape[-1]
        # Add some action noise for exploration
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        # Because we use parameter noise, we should use a MlpPolicy with layer normalization
        model = TD3('MlpPolicy', env, action_noise=action_noise, verbose=0)
    elif index == 3:
        model = TRPO("MlpPolicy", env, verbose=0)

    return learners[index].lower(), model

def create_savings(env_id, learner_name):
    log_dir = "model/{}".format(env_id)
    result_dir = "results/{}/{}".format(env_id, learner_name)
    tools.mkdir(log_dir)
    tools.mkdir(result_dir)
    return log_dir, result_dir

if __name__ == '__main__':
    tools.display_torch_device()
    load_model = True
    train = False
    # tools.set_logger_level(1)
    for i in range(1, len(env_list)):
        process_monitor = monitor.Process_Monitor()
        env_id = get_environments(i)
        logger.warning('train {} env'.format(env_id))
        env = gym.make(env_id)

        for learner_index in range(len(learners)):
            learner_name, model = get_model(index=learner_index, env=env)
            log_dir, result_dir = create_savings(env_id=env_id, learner_name=learner_name)

            logger.warning('using {} model'.format(learner_name))

            if load_model:
                model = model.load(os.path.join(log_dir, '{}_best.zip'.format(learner_name)))
                _, reward = monitor.evaluate_policy(env=env, model=model)
                logger.success('loading model with rewards {}'.format(reward))

            train_steps = 0
            max_train_steps = 200
            best_reward = float('-inf')
            while train and train_steps < max_train_steps:
                train_steps += 1
                logger.success('train {} {} times - left {} steps'.format(learner_name, train_steps, max_train_steps-train_steps))
                model.learn(total_timesteps=1000)
                _, reward = monitor.evaluate_policy(env=env, model=model)
                process_monitor.store(reward=reward)
                average_reward = process_monitor.average(50)
                '''
                if average_reward > reward_threshold:
                    tools.save_video(env_id, model, result_dir)
                '''
                if reward > best_reward:
                    best_reward = reward
                    model.save(os.path.join(log_dir, '{}_best.zip'.format(learner_name)))
                    logger.success('update best model')

                if train_steps % 10 == 0:
                    model.save(os.path.join(log_dir, '{}_tmp.zip'.format(learner_name)))
            
            if train:
                process_monitor.plot_learning_curve('{}/{}'.format(result_dir, learner_name))
                process_monitor.plot_average_learning_curve('{}/{}'.format(result_dir, learner_name), 50)

            tools.save_video(env_id, model, learner_name, result_dir) 
            