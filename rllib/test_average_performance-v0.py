"""
Compute success ratio by using learned model
"""
import pygame
import os
import argparse
import numpy as np
import tensorflow as tf
import gym
from myenv import MyEnv
import ray
import ray.rllib.agents.ppo as ppo

N_EVAL_EPISODES = 1000

ALGORITHM = 'ppo'
NUM_WORKERS = 8

# ENV_TYPE = 'rl'
# MODEL_NAME = 'myenv-v0-ppo/checkpoints/trial_6_best/checkpoint_341/checkpoint-341'
ENV_TYPE = 'gan_rl'
MODEL_NAME = 'myenv-v0-ppo/checkpoints/trial_7_best/checkpoint_411/checkpoint-411'


def main():
    # Initialize ray
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    # Generate & Check environment
    env = MyEnv({})

    # Define trainer agent
    model_name = MODEL_NAME

    config = ppo.DEFAULT_CONFIG.copy()
    config['env_config'] = {}
    config['num_workers'] = NUM_WORKERS
    config['num_gpus'] = 0
    config['framework'] = 'tfe'
    config['eager_tracing'] = True

    agent = ppo.PPOTrainer(config=config, env=MyEnv)
    agent.restore(model_name)

    success_history = []
    success_count = 0
    for idx in range(N_EVAL_EPISODES):
        """ Initialization """
        observation = env.reset()
        frames = []

        """ Save some initial values """
        fighter_0 = env.fighter.ingress
        jammer_0 = env.jammer.ingress

        while True:
            action_index = agent.compute_action(observation)

            # 環境を１step 実行
            observation, reward, done, info = env.step(action_index)

            # 環境の描画とビデオ録画
            # shot = env.render(mode=args.render_mode)
            frames.append(env.render(mode=args.render_mode))

            # Slow down rendering
            # pygame.time.wait(10)

            # エピソードの終了処理
            if done:
                success_history.append(info['success'])
                if info['success'] > .5:
                    success_count += 1
                break

    n_success = success_count
    n_fail = N_EVAL_EPISODES - n_success
    if np.sum(success_history) != success_count:
        raise Exception('Something is wrong!')

    """ Summarize results """
    print('==================== Summary of the results ====================')
    print(f'Mission conditions = w1 : w2 : w3 = '
          f'{env.mission_probability[0]:.3f} : {env.mission_probability[1]:.3f} : {env.mission_probability[2]:.3f}')
    print(f'   Model is < {MODEL_NAME} >')
    print(f'   Number of success missions: {round(n_success)} / {N_EVAL_EPISODES},  '
          f'   Number of failed missions {round(n_fail)} / {N_EVAL_EPISODES}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_mode", type=str, default="human", help="rgb, if training")
    args = parser.parse_args()

    main()
