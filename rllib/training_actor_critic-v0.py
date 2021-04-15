"""
rllib version of ppo combined with GAN planner
Need 'myenv.py' in the same directory, because rllib handling of custom environment
This 'myenv.py' is different version from stable_baselines version
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from myenv import MyEnv

import ray
from ray.tune.logger import Logger, UnifiedLogger
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print

import tensorflow as tf
import numpy as np
import datetime
import os
import tempfile

PROJECT = 'myenv-v0-ppo'
ID = 7
TRIAL = 'trial_' + str(ID)
NUM_WORKERS = 8
MAX_EPISODE = 10000
N_EVAL_EPISODE = 1000
EVAL_FREQ = 10


def custom_log_creator(custom_path, custom_str):
    timestr = datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def main():
    # Initialize ray
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    # Define trainer agent
    config = ppo.DEFAULT_CONFIG.copy()
    config['env_config'] = {}
    config['num_gpus'] = 0
    config['num_workers'] = NUM_WORKERS
    config['num_cpus_per_worker'] = 1
    config['framework'] = 'tfe'
    config['eager_tracing'] = True
    # config['model']['fcnet_hiddens'] = [64, 64, 64]
    print(pretty_print(config))
    trainer = ppo.PPOTrainer(config=config,
                             env=MyEnv,
                             logger_creator=custom_log_creator(
                                 os.path.expanduser("./" + PROJECT + "/logs"), TRIAL))

    logdir = trainer.logdir
    print(f'\n********************** logdir = {logdir}\n')

    # Check trainer agent
    policy = trainer.get_policy()
    policy.model.base_model.summary()

    # Define evaluator agent
    eval_env = MyEnv({})
    # obs = eval_env.reset()

    # Train agent
    max_episode = MAX_EPISODE
    eval_freq = EVAL_FREQ
    n_eval_episode = N_EVAL_EPISODE
    best_success_count = -100
    best_checkpoint_dir = os.path.join('./' + PROJECT + '/checkpoints/', TRIAL + '_best')
    success_history = []
    iteration_history = []
    for i in range(max_episode):
        print(f'{i}th iteration is starting.')
        # Training
        result = trainer.train()
        # print(pretty_print(result))

        # Evaluation
        if i % eval_freq == 0:
            print(f'\n--------------- Evaluation results at {i}th iteration ---------------')
            print(pretty_print(result))
            total_return = 0
            success_count = 0
            info = {}
            return_list = []
            for j in range(n_eval_episode):
                # Test the trained agent
                obs = eval_env.reset()
                done = False

                while not done:
                    action = trainer.compute_action(obs)
                    obs, reward, done, info = eval_env.step(action)
                    total_return += reward

                return_list.append(total_return)
                if info['success'] > 0.5:
                    success_count += 1

            print(f'\niteration {i} success_count: {success_count} / {n_eval_episode}')
            # print(f'return list: {return_list}')

            success_history.append(success_count / n_eval_episode)
            iteration_history.append(i)

            success_history_np = np.array(success_history)
            file_name = './' + PROJECT + '/learning_history/trial_' + str(ID)
            # np.savez('./learning_history', iteration_history, success_history)
            np.savez(file_name, iteration_history, success_history)

            if success_count >= best_success_count:
                best_checkpoint = trainer.save(checkpoint_dir=best_checkpoint_dir)
                print(f'best checkpoint saved at {best_checkpoint}\n')
                best_success_count = success_count

            print(f'------------------------------------------------------------\n')


if __name__ == '__main__':
    main()
