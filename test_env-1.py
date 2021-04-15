"""
Testing code for env.MyEnv
"""
import pygame
import os
import argparse
import numpy as np
import tensorflow as tf
import gym
import myenv_1


def pause_for_debug():
    pause = False
    for event in pygame.event.get():
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                pause = True

    while pause:
        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    pause = False


def status_print(env, observation, reward, done):
    # print(f'action_index: {action_index}: '
    #      f'fighter.action: {env.fighter.action},   jammer.action: {env.jammer.action}')
    print(f'sam.offset: {env.sam.offset:.1f},   fighter.ingress: {env.fighter.ingress:.1f},   '
          f'jammer.ingress: {env.jammer.ingress:.1f}')

    f_ingress = observation[0] * env.sam.max_offset
    f_range = observation[1] * env.fighter.max_firing_range
    j_ingress = observation[2] * env.sam.max_offset
    s_offset = observation[3] * env.sam.max_offset
    s_f_range = observation[4] * env.sam.max_firing_range
    s_jammed_f_range = observation[5] * env.sam.max_firing_range * env.jammer.jam_effectiveness

    c1 = f_ingress + f_range
    c2 = j_ingress + env.jammer.jam_range
    c3 = s_offset - s_f_range
    c4 = s_offset - s_jammed_f_range

    print(f'   obs: fighter.ingress: {f_ingress:.1f},   fighter.f_range: {f_range:.1f},'
          f'   jammer.ingress: {j_ingress: .1f},   sam.offset: {s_offset: .1f}')
    print(f'   obs: sam.f_range: {s_f_range:.1f},   sam.jammed_f_range: {s_jammed_f_range:.1f},'
          f'   sam.alive: {observation[6]:.0f},   jammer.on: {observation[7]:.0f}, '
          f'   done: {done},   reward: {reward}')

    if done:
        print(f'\n*************** Mission Condition: {env.mission_condition} ***************')
        print(f'   fighter.firing_range: {env.fighter.firing_range:.1f} km')
        print(f'   sam.firing_range: {env.sam.firing_range:.1f} km')
        print(f'   sam.jammed_firing_range: {env.sam.jammed_firing_range:.1f} km')
        print(f'   sam.off_set: {env.sam.offset:.1f} km')

        print(f'\nfighter.alive: {env.fighter.alive},   jammer.alive: {env.jammer.alive},'
              f'   jammer.on: {env.jammer.on},   sam.alive: {env.sam.alive},  reward: {reward}')

        if (env.sam.alive < .5) and (env.fighter.alive > .5) and (env.jammer.alive > .5) \
                and (env.jammer.on < 0.5):
            print('\nBlue win without using Jammmer -------------------')
            print(f'   Mission condition: {env.mission_condition} ')
            print(f'   obs results: fighter.ingress + fighter.firing_range: {c1:.1f}')
            print(f'                jammer.ingress + jammer.jam_range: {c2:.1f}')
            print(f'                sam.offset: {s_offset:.1f}')
            print(f'                sam.offset - sam.firing_range: {c3:.1f}')
            # print(f'                sam.offset - sam.jammed_firing_range: {c4:.1f}')
            print(f'                fighter.ingress: {f_ingress:.1f}')
            print(f'                jammer.ingress: {j_ingress:.1f}')

            print('\nFor win')
            print(f'   fighter.ingress + fighter.firing_range: {c1:.1f} >= sam.offset: {s_offset:.1f}')
            print(f'   sam.offset - sam.firing_range: {c3:.1f} > fighter.ingress: {f_ingress:.1f}')
            print(f'   sam.offset - sam.firing_range: {c3:.1f} > jammer.ingress: {j_ingress:.1f}')

        elif (env.sam.alive < .5) and (env.fighter.alive > .5) and (env.jammer.alive > .5) \
                and (env.jammer.on > 0.5):
            print('\nBlue win with using Jammer --------------------------')
            print(f'   Mission condition: {env.mission_condition} ')
            print(f'   obs results: fighter.ingress + fighter.firing_range: {c1:.1f}')
            print(f'                jammer.ingress + jammer.jam_range: {c2:.1f}')
            print(f'                sam.offset: {s_offset:.1f}')
            # print(f'                sam.offset - sam.firing_range: {c3:.1f}')
            print(f'                sam.offset - sam.jammed_firing_range: {c4:.1f}')
            print(f'                fighter.ingress: {f_ingress:.1f}')
            print(f'                jammer.ingress: {j_ingress:.1f}')

            print('\nFor win')
            print(f'   fighter.ingress + fighter.firing_range: {c1:.1f} >= sam.offset: {s_offset:.1f}')
            print(f'   sam.offset - sam.jammed_firing_range: {c4:.1f} > fighter.ingress: {f_ingress:.1f}')
            print(f'   sam.offset - sam.jammed_firing_range: {c4:.1f} > jammer.ingress: {j_ingress:.1f}')

        else:
            print('\nBlue loose -------------------------')
            print(f'   Mission condition: {env.mission_condition} ')
            print(f'   obs results: sam.offset - sam.firing_range: {c3:.1f}')
            print(f'                sam.offset - sam.jammed_firing_range: {c4:.1f}')
            print(f'                fighter.ingress: {f_ingress:.1f}')
            print(f'                jammer.ingress: {j_ingress:.1f}')
            print(f'                fighter.ingress + fighter.firing_range: {c1:.1f}')
            print(f'                jammer.ingress + jammer.jam_range: {c2:.1f}')
            print(f'                sam.offset: {s_offset:.1f}')

            print('\nFor win')
            print(f'   fighter.ingress + fighter.firing_range: {c1:.1f} >= sam.offset: {s_offset:.1f}')
            if env.jammer.on < 0.5:
                print(f'   sam.offset - sam.firing_range: {c3:.1f} > fighter.ingress: {f_ingress:.1f}')
                print(f'   sam.offset - sam.firing_range: {c3:.1f} > jammer.ingress: {j_ingress:.1f}')
            else:
                print(f'   sam.offset - sam.jammed_firing_range: {c4:.1f} > fighter.ingress: {f_ingress:.1f}')
                print(f'   sam.offset - sam.jammed_firing_range: {c4:.1f} > jammer.ingress: {j_ingress:.1f}')

        if (env.fighter.ingress < env.fighter.allowable_negative_ingress) or \
                (env.jammer.ingress < env.jammer.allowable_negative_ingress):
            print('Negative ingress -------------------------')


def count_w_and_success(env, w_count, success_count, reward_count, reward, idx):
    w_count[idx] += 1
    if (env.sam.alive < .5) and (env.fighter.alive > .5) and (env.jammer.alive > .5):
        success_count[idx] += 1
    reward_count[idx] += reward
    return w_count, success_count, reward_count


def conunt_results(env, w_count, success_count, reward_count, reward):
    w_id = env.mission_condition
    if w_id == "w1":
        idx = 0
    elif w_id == "w2":
        idx = 1
    elif w_id == "w3":
        idx = 2
    elif w_id == "l1":
        idx = 3
    elif w_id == "l2":
        idx = 4
    else:
        raise Exception('Error!')

    w_count, success_count, reward_count = \
        count_w_and_success(env, w_count, success_count, reward_count, reward, idx)
    return w_count, success_count, reward_count


def main(w_count, success_count, reward_count):
    env = gym.make('myenv-v1')

    while True:
        # print(f'step {step}')
        # ランダムアクションの選択
        # action_index = env.action_space.sample()

        a = np.array([0, 1, 2, 3])
        p = np.array([1, 1, 1, 1])
        """
        a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        p = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        """
        p = p / np.sum(p)
        action_index = np.random.choice(a, p=p)

        # 環境を１step 実行
        observation, reward, done, _ = env.step(action_index)
        if args.render_mode == 'human':
            print(f'\naction is selected at {env.steps}')
            status_print(env, observation, reward, done)

        # 環境の描画
        shot = env.render(mode=args.render_mode)

        # Space keyでpause, デバッグ用
        pause_for_debug()

        # エピソードの終了処理
        if done:
            # print('done')
            w_count, success_count, reward_count = \
                conunt_results(env, w_count, success_count, reward_count, reward)
            break
    return w_count, success_count, reward_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_mode", type=str, default="rgb", help="rgb, if training")
    args = parser.parse_args()

    w_count = [0, 0, 0, 0, 0]
    success_count = [0, 0, 0, 0, 0]
    reward_count = [0, 0, 0, 0, 0]

    for _ in range(100000):
        w_count, success_count, reward_count = main(w_count, success_count, reward_count)

    print('\n\n------- Summaries of Missions by random actions -------')
    print(f'   Success: {np.sum(success_count)} / {np.sum(w_count)},   '
          f'{np.sum(success_count) / np.sum(w_count) * 100} [%]')
    print(f'   Breakdown of Missions: {w_count}')
    print(f'   Breakdown of Success:  {success_count},    '
          f'ratio: {success_count / np.sum(w_count) * 100} [%],   '
          f'reward: {reward_count}')
