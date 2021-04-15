"""
Testing code for env.MyEnv
"""
import pygame
import argparse
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import gym
import myenv
import imageio
import json
import os
from stable_baselines import PPO2, A2C

ALGORITHM = 'ppo2'  # 'ppo2' or 'a2c'
ENV_TYPE = 'ppo2_8'
MODEL_NAME = 'myenv-v0-ppo2/' + ENV_TYPE + '/best_model.zip'

# ALGORITHM = 'a2c'  # 'ppo2' or 'a2c'
# ENV_TYPE = 'a2c_8'
# MODEL_NAME = 'myenv-v0-a2c/' + ENV_TYPE + '/best_model.zip'

video_dir = './videos/' + ENV_TYPE + '/'


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


def status_print(env, observation, reward, done, fighter_0, jammer_0):
    # print(f'action_index: {action_index}: '
    #      f'fighter.action: {env.fighter.action},   jammer.action: {env.jammer.action}')
    print(f'fighter_initial_ingress: {fighter_0} km,   jammer_initial_ingress: {jammer_0} km')
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


def count_w_and_success(env, w_count, success_count, idx):
    w_count[idx] += 1
    if (env.sam.alive < .5) and (env.fighter.alive > .5) and (env.jammer.alive > .5):
        success_count[idx] += 1
    return w_count, success_count


def conunt_results(env, w_count, success_count):
    w_id = env.mission_condition
    if w_id == "w1":
        idx = 0
    elif w_id == "w2":
        idx = 1
    elif w_id == "w3":
        idx = 2
    else:
        raise Exception('Error!')

    w_count, success_count = count_w_and_success(env, w_count, success_count, idx)
    return w_count, success_count


def make_video(video_name, frames):
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)
    filename = video_dir + video_name + '.gif'
    # imageio.mimsave(filename, np.array(frames), 'GIF', fps=10)
    imageio.mimsave(filename, np.array(frames), fps=30)


def make_jason(env, video_name, fighter_0, jammer_0):
    filename = video_dir + video_name + '.json'

    results = dict()
    if (env.sam.alive < .5) and (env.fighter.alive > .5) and (env.jammer.alive > .5) \
            and (env.jammer.on < 0.5):
        results['Mission'] = 'Success without using Jammer'
    elif (env.sam.alive < .5) and (env.fighter.alive > .5) and (env.jammer.alive > .5) \
            and (env.jammer.on > 0.5):
        results['Mission'] = 'Success with using Jammer'
    else:
        results['Mission'] = 'Failed'

    results['mission_condition'] = env.mission_condition
    results['fighter'] = {'alive': env.fighter.alive,
                          'initial_ingress': fighter_0,
                          'ingress': env.fighter.ingress,
                          'firing_range': env.fighter.firing_range}

    results['jammer'] = {'alive': env.jammer.alive,
                         'initial_ingress': jammer_0,
                         'ingress': env.jammer.ingress,
                         'jamming': env.jammer.on}

    results['sam'] = {'alive': env.sam.alive,
                      'offset': env.sam.offset,
                      'firing_range': env.sam.firing_range,
                      'jammed_firing_range': env.sam.jammed_firing_range}

    # print(json.dumps(results, ensure_ascii=False, indent=2))

    with open(filename, mode='wt', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=2)


def main():
    model_dir = './models'
    model_name = model_dir + '/' + MODEL_NAME

    """ Generate & Check environment """
    env_name = 'myenv-v0'
    env = gym.make(env_name)
    # env = gym.wrappers.Monitor(env, "./videos", force=True)  # For video making

    """ Vectorize environment """
    # Unnecessary to vectorize environment
    # env = DummyVecEnv([lambda: env])

    """ Load model and set environment """
    if ALGORITHM == 'ppo2':
        model = PPO2.load(model_name)
    elif ALGORITHM == 'a2c':
        model = A2C.load(model_name)
    else:
        raise Exception('Load error.  Specify proper name')

    for idx in range(90):
        """ Initialization """
        observation = env.reset()
        frames = []

        """ Save some initial values """
        fighter_0 = env.fighter.ingress
        jammer_0 = env.jammer.ingress

        while True:
            action_index, _ = model.predict(observation)

            # 環境を１step 実行
            observation, reward, done, _ = env.step(action_index)

            # 環境の描画とビデオ録画
            # shot = env.render(mode=args.render_mode)
            frames.append(env.render(mode=args.render_mode))

            # Space keyでpause, デバッグ用
            pause_for_debug()

            # Slow down rendering
            pygame.time.wait(10)

            # エピソードの終了処理
            if done:
                status_print(env, observation, reward, done, fighter_0, jammer_0)
                video_name = ALGORITHM + '_' + env.mission_condition + '-' + str(idx)
                make_video(video_name, frames)
                make_jason(env, video_name, fighter_0, jammer_0)
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_mode", type=str, default="human", help="rgb, if training")
    args = parser.parse_args()

    main()
