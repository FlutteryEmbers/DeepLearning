from utils import monitor

process_monitor = monitor.Process_Monitor()
process_monitor.load_from_file('results/LunarLander-v2/dqn/dqn_history_rewards.pickle')
print(process_monitor.rewards)