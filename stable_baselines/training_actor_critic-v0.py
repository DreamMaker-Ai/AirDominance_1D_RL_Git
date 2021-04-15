"""
Stable baselines: Actor-Critic Algorithm - PPO2 & A2C
    Stable Baselines require tensorflow 1
        https://stable-baselines.readthedocs.io/en/master/guide/install.html
    Require installing myenv by "pip install -e myenv"
"""
import gym
import os
import tensorflow as tf
import pickle
import myenv
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2, A2C
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback

# from stable_baselines.bench import Monitor

PROJECT = 'Rand report 1D RL'  # For wandb
CONTINUAL_LEARNING = False

ENV_NAME = 'myenv-v0'
ALGORITHM = 'a2c'  # 'ppo2' or 'a2c'
NUM_ENVS = 4  # Use multiple environments
MODEL_NAME = ALGORITHM + '_' + str(NUM_ENVS)
LOAD_MODEL_NAME = MODEL_NAME
MAX_STEPS = 1000000 * NUM_ENVS
N_EVAL_EPISODES = 1000
EVAL_FREQ = 10000
SAVE_FREQ = 100000

DEFAULT = True
if not DEFAULT:
    ACT_FUN = tf.nn.relu
    # NET_ARCH = [64, dict(vf=[64, 64], pi=[64])]
    NET_ARCH = [dict(vf=[64, 64, 64], pi=[64, 64, 64])]
else:
    ACT_FUN = 'default'
    NET_ARCH = 'default'


def make_root_log_dirs():
    # Make log directory
    if not os.path.exists('./logs'):
        os.mkdir('./logs')

    root_logs_dir = './logs'
    if not os.path.exists(root_logs_dir):
        os.mkdir(root_logs_dir)
    return root_logs_dir


def make_root_save_dirs():
    if not os.path.exists('./models'):
        os.mkdir('./models')

    root_save_dir = "./models"
    if not os.path.exists(root_save_dir):
        os.mkdir(root_save_dir)
    return root_save_dir


def prepare_dirs():
    """ Prepare for trainings """
    log_dir = os.path.join(root_logs_dir, ENV_NAME + '-' + ALGORITHM)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    model_dir = os.path.join(root_save_dir, ENV_NAME + '-' + ALGORITHM)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    return log_dir, model_dir  # ./logs/myenv-v0-ppo2, ./models/myenv-v0-ppo2


def summarize_config(env):
    config = {'env_name': ENV_NAME,
              'n_eval_episodes': N_EVAL_EPISODES,
              'observation': env.observation_space,
              'action': env.action_space,
              'algorithm': ALGORITHM,
              'continual_learning': CONTINUAL_LEARNING,
              'model_name': MODEL_NAME,
              'num_envs': NUM_ENVS,
              'net_arch': NET_ARCH,
              'mission_conditions': env.mission_conditions,
              'mission_probability': env.mission_probability}
    print('\n--------------- Summary of configs ---------------')
    for k, v in config.items():
        print(f'{k}: {v}')
    print('\n')

    return config


def define_model(env, log_dir):
    if DEFAULT:
        policy_kwargs = dict()
    else:
        policy_kwargs = dict(act_fun=ACT_FUN, net_arch=NET_ARCH)

    if ALGORITHM == 'ppo2':
        model = PPO2(policy=MlpPolicy,
                     env=env,
                     policy_kwargs=policy_kwargs,
                     verbose=0,
                     tensorboard_log=log_dir)

    elif ALGORITHM == 'a2c':
        model = A2C(policy=MlpPolicy,
                    env=env,
                    policy_kwargs=policy_kwargs,
                    verbose=0,
                    tensorboard_log=log_dir)
    else:
        raise Exception('Specify proper algorithm')

    model_arch = model.get_parameter_list()
    print('\n--------------- Summary of archs ---------------')
    for model_param in model_arch:
        print(model_param)
    print('\n')

    return model


def load_model(env, model_dir, log_dir):
    model_name = model_dir + '/' + LOAD_MODEL_NAME + '.zip'
    print(f'----- model will be loaded from {model_name} \n')

    """ Load trained model, then continue training """
    if ALGORITHM == 'ppo2':
        model = PPO2.load(model_name, verbose=0, tensorboard_log=log_dir)
    elif ALGORITHM == 'a2c':
        model = A2C.load(model_name, verbose=0, tensorboard_log=log_dir)
    else:
        raise Exception('Specify Algorithm')

    model.set_env(env)

    return model


def save_config(log_dir, config):
    config_file = log_dir + '/config.pickle'
    print(f'* Config file is save as {config_file}\n')

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    with open(config_file, mode='wb') as f:
        pickle.dump(config, f)


def main():
    """ Prepare for trainings """
    log_dir, model_dir = prepare_dirs()

    model_name = model_dir + '/' + MODEL_NAME
    print(f'model will be saved as {model_name}')

    log_dir = log_dir + '/' + MODEL_NAME

    """ Generate & Check environment """
    env_name = ENV_NAME
    env = gym.make(env_name)
    # print(f'Observation space: {env.observation_space}')
    # print(f'Action space: {env.action_space}')
    # env = Monitor(env, log_dir, allow_early_resets=True)
    # check_env(env)

    """ Save config as pickle file """
    config = summarize_config(env)
    save_config(log_dir, config)

    """ Vectorize environment """
    num_envs = NUM_ENVS
    env = DummyVecEnv([lambda: env for _ in range(num_envs)])  # For training

    eval_env = DummyVecEnv([lambda: gym.make(env_name)])  # For evaluation

    """ Define checkpoint callback """
    checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ,
                                             save_path=model_name,
                                             name_prefix=MODEL_NAME)

    """ Use deterministic actions for evaluation callback """
    eval_callback = EvalCallback(eval_env, best_model_save_path=model_name,
                                 log_path=log_dir, eval_freq=EVAL_FREQ,
                                 deterministic=True, render=False,
                                 n_eval_episodes=N_EVAL_EPISODES)

    print(f'Algorithm: {ALGORITHM}\n')

    if not CONTINUAL_LEARNING:
        """ Define model """
        model = define_model(env, log_dir)
    else:
        model = load_model(env, model_dir, log_dir)

    """ Evaluate model before training """
    # mean_reward, std_reward = evaluate_policy(model=model,
    #                                          env=eval_env,
    #                                          n_eval_episodes=N_EVAL_EPISODES)
    # print(f'Before training: mean reward: {mean_reward:.2f} +/- {std_reward:.2f}')

    """ Train model """
    model.learn(total_timesteps=MAX_STEPS,
                callback=[checkpoint_callback, eval_callback])

    """ Evaluate model after training """
    # mean_reward, std_reward = evaluate_policy(model=model,
    #                                          env=eval_env,
    #                                          n_eval_episodes=N_EVAL_EPISODES)
    # print(f'After training: mean reward: {mean_reward:.2f} +/- {std_reward:.2f}')

    """ Save trained model """
    model.save(model_name)

    """ Test trained model """
    obs = eval_env.reset()
    for i in range(N_EVAL_EPISODES):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = eval_env.step(action)
        eval_env.render()

    env.close()
    eval_env.close()


if __name__ == '__main__':
    # Make log directory
    root_logs_dir = make_root_log_dirs()

    # Make save directory
    root_save_dir = make_root_save_dirs()

    main()
