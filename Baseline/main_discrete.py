import gym, os
from loguru import logger
import numpy as np

from stable_baselines3 import TD3, SAC, PPO, DQN
from sb3_contrib import TRPO

from stable_baselines3.common.noise import NormalActionNoise

from utils import tools, monitor
from loguru import logger
# import mujoco_py
# mj_path = mujoco_py.utils.discover_mujoco()

load_model = False
train = True
reward_decay = True

learners = ['DQN', 'PPO', 'TRPO']
env_list = ["LunarLander-v2"]
max_train_steps = 3
# reward_threshold = []

def get_environments(choice):
    logger.critical('Trainning: {}'.format(env_list[choice]))
    return env_list[choice]

def get_model(index, env):
    if  index == 0:
        model = DQN("MlpPolicy", env, verbose=0)
    elif index == 1:
        model = PPO("MlpPolicy", env, verbose=0)
    elif index == 2:
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
    # tools.set_logger_level(1)
    for i in range(1):
        env_id = get_environments(i)
        logger.warning('env {}'.format(env_id))
        env = gym.make(env_id)
        env.seed(10)

        for learner_index in range(len(learners)):
            model = None
            learner_name, model = get_model(index=learner_index, env=env)
            log_dir, result_dir = create_savings(env_id=env_id, learner_name=learner_name)

            logger.warning('using {} model'.format(learner_name))

            if load_model:
                model = model.load(os.path.join(log_dir, '{}_best.zip'.format(learner_name)))
                _, reward = monitor.evaluate_policy(env=env, model=model)
                logger.success('loading model with rewards {}'.format(reward))

            train_steps = 0
            best_reward = float('-inf')
            process_monitor = monitor.Process_Monitor(output_dir=result_dir, name=learner_name)

            while train and train_steps < max_train_steps:
                train_steps += 1
                logger.success('trained in {} using {} {} times - left {} steps'.format(env_id, learner_name, train_steps, max_train_steps-train_steps))
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
                process_monitor.plot_learning_curve()
                process_monitor.plot_average_learning_curve(50)
                process_monitor.dump_to_file()
                process_monitor.reset()

            tools.save_video(env_id, model, learner_name, result_dir) 
            