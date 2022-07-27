from utils import monitor
import numpy as np
import matplotlib.pyplot as plt
from utils import tools
import math

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

mode = 'discrete'
env_list = config[mode]['env_list']
trainers = config[mode]['trainers']

n = 50

def get_average(rewards, n):
    np_reward = np.array(rewards)
    means = np.zeros(len(rewards))
    x = np.arange(len(rewards))
    for i in range(len(means)):
        means[i] = np.mean(np_reward[max(0, i-n):(i+1)])
    return x, means

for env_id in env_list:
    filename = '{} average_reward_per_{}.png'.format(env_id, n)
    for trainer in trainers:
        name = trainer.lower()
        process_monitor = monitor.Process_Monitor()
        process_monitor.load_from_file('results/{}/{}/{}_history_rewards.pickle'.format(env_id, name, name))
        # print(process_monitor.rewards)
        max_reward = max(process_monitor.rewards)
        '''
        for i in range(len(process_monitor.rewards)):
            if math.floor(process_monitor.rewards[i]) == math.floor(-428.79):
                print(i)
        '''
        print('{} {} has max reward of {}'.format(env_id, name, max_reward))
        x, means = get_average(process_monitor.rewards, n) 
        plt.plot(x, means, label=name)
    
    plt.legend(loc='upper left')
    plt.title(label='{} average_reward_per_{}'.format(env_id, n))
    plt.savefig(filename)
    plt.close()

