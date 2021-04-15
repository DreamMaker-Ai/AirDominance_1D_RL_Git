"""
Compute success ratio by using learned model
"""
import pygame
import os
import argparse
import numpy as np
import tensorflow as tf
import gym
import myenv
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.vec_env import VecVideoRecorder
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import PPO2, A2C, DQN

N_EVAL_EPISODES = 1000

# ALGORITHM = 'ppo2'  # 'ppo2' or 'a2c'
# MODEL_NAME = 'myenv-v0-ppo2/ppo2_8/best_model.zip'

# ALGORITHM = 'a2c'  # 'ppo2' or 'a2c'
# MODEL_NAME = 'myenv-v0-a2c/a2c_8/best_model.zip'

ALGORITHM = 'dqn'  # 'ppo2' or 'a2c'
MODEL_NAME = 'myenv-v0-dqn/dqn_1/best_model.zip'


def main():
    model_dir = './models/'
    model_name = model_dir + MODEL_NAME

    """ Generate & Check environment """
    env_name = 'myenv-v0'
    env = gym.make(env_name)
    # env = Monitor(env, 'logs')
    # check_env(env)
    mission_probability = env.mission_probability

    """ Vectorize environment """
    env = DummyVecEnv([lambda: env])

    """ Load model and set environment """
    if ALGORITHM == 'ppo2':
        model = PPO2.load(model_name)
    elif ALGORITHM == 'a2c':
        model = A2C.load(model_name)
    elif ALGORITHM == 'dqn':
        model = DQN.load(model_name)
    else:
        raise Exception('Load error.  Specify proper name')

    """ Perform simulaion    """
    mean_reward, std_reward = evaluate_policy(model=model, env=env, n_eval_episodes=N_EVAL_EPISODES)
    n_success = mean_reward * N_EVAL_EPISODES
    n_fail = N_EVAL_EPISODES - n_success

    """ Summarize results """
    print('==================== Summary of the results ====================')
    print(f'Mission conditions = w1 : w2 : w3 = '
          f'{mission_probability[0]:.3f} : {mission_probability[1]:.3f} : {mission_probability[2]:.3f}')
    print(f'   Model is < {MODEL_NAME} >')
    print(f'   Number of success missions: {round(n_success)} / {N_EVAL_EPISODES},  '
          f'   Number of failed missions {round(n_fail)} / {N_EVAL_EPISODES}')
    print(f'   Success ratio: {mean_reward:.2f} +/- {std_reward:.2f}')
    print(f'   Success percentage: {mean_reward * 100:.2f} %')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_mode", type=str, default="human", help="rgb, if training")
    args = parser.parse_args()

    main()
