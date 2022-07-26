import torch, random, yaml, sys, os, re
import numpy as np
from loguru import logger
import gym
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from tqdm import tqdm

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def render(env, model):
    obs = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

def load_config(file):
    logger.debug('loading {}'.format(file))
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    with open(file, 'r') as stream:
        config = yaml.load(stream, Loader=loader)

    if config == None:
        sys.exit('{} did not loaded correctly'.format(file))
        
    return config

def set_logger_level(level):
    choice = ['Trace', 'DEBUG', 'Info', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL']
    logger.remove()
    # logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}", level=choice[level])
    logger.add(sys.stderr, level=choice[level])

def mkdir(dir):
    isExist = os.path.exists(dir)
    if not isExist:
        os.makedirs(dir)

def get_environments(choice):
    env_list = ["LunarLanderContinuous-v2", 'HalfCheetah-v2', 'Hopper-v2']
    logger.critical('Trainning: {}'.format(env_list[choice]))
    return env_list[choice]

def save_video(env_id, model, model_name, video_folder, video_length=10000):
    logger.warning('Making Video for {}'.format(env_id))
    env = DummyVecEnv([lambda: gym.make(env_id)])
    # if video_length == None:
    #     video_length = render_once(env, model)

    obs = env.reset()
    # Record the video starting at the first step
    env = VecVideoRecorder(env, video_folder,
                        record_video_trigger=lambda x: x == 0, video_length=video_length,
                        name_prefix=model_name)

    env.reset()
    for i in tqdm(range(video_length), desc="Video Frame Used"):
        action, _state = model.predict(obs, deterministic=True)
        # action = [env.action_space.sample()]
        obs, reward, done, info = env.step(action)
        if done:
            break
    # Save the video
    env.close()
    logger.success('Video Saved')

def display_torch_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)