"""
Testing code for env.MyEnv
"""
import pygame
import argparse
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import gym
import myenv_2
import imageio
import json
from stable_baselines import PPO2, A2C

ALGORITHM = 'ppo2'  # 'ppo2' or 'a2c'
MODEL_NAME = 'myenv-v2-ppo2/ppo2_8/best_model.zip'
NUM_TRIALS = 100

video_dir = './videos/myenv-v2-ppo2/'


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
    print(f'\n\n*************** Mission Summaries: {env.mission_condition} ***************')
    if reward > 0:
        print(f'Mission Succeeded for mission condition {env.mission_condition}')
    elif reward < 0:
        print('Mission Failed for mission condition {env.mission_condition}')
    else:
        raise Exception('Something is wrong')

    if env.jammer.on > 0.5:
        print('   Jammer is used')
    else:
        print('   Jammer is not used')

    print(f'\n--------------- Mission Conditions: {env.mission_condition} ---------------')
    print(f'   fighter.firing_range: {env.fighter.firing_range:.1f} km')
    print(f'   sam.firing_range: {env.sam.firing_range:.1f} km')
    print(f'   sam.jammed_firing_range: {env.sam.jammed_firing_range:.1f} km')
    print(f'   sam.off_set: {env.sam.offset:.1f} km')

    print(f'\n--------------- Initial Conditions: {env.mission_condition} ---------------')
    print(f'fighter_initial_ingress: {fighter_0} km,   jammer_initial_ingress: {jammer_0} km')

    print(f'\n--------------- End status: {env.mission_condition} ---------------')
    print(f'fighter.ingress: {env.fighter.ingress:.1f} km,   jammer.ingress: {env.jammer.ingress:.1f} km')
    print(f'fighter.alive: {env.fighter.alive},   jammer.alive: {env.jammer.alive},   '
          f'jammer.on: {env.jammer.on},   sam.alive: {env.sam.alive}')

    print(f'\n--------------- End observation: {env.mission_condition} ---------------')
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

    print(f'obs: fighter.ingress: {f_ingress:.1f},   fighter.f_range: {f_range:.1f},'
          f'   jammer.ingress: {j_ingress: .1f}')
    print(f'sam.offset: {s_offset: .1f},   obs: sam.f_range: {s_f_range:.1f},'
          f'   sam.jammed_f_range: {s_jammed_f_range:.1f},')
    print(f'sam.alive: {observation[6]:.0f},   jammer.on: {observation[7]:.0f}, '
          f'   done: {done},   reward: {reward}')

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
    elif w_id == "l1":
        idx = 3
    elif w_id == "l2":
        idx = 4
    else:
        raise Exception('Error!')

    w_count, success_count = count_w_and_success(env, w_count, success_count, idx)
    return w_count, success_count


def make_video(video_name, frames):
    filename = video_dir + video_name + '.gif'
    # imageio.mimsave(filename, np.array(frames), 'GIF', fps=10)
    imageio.mimsave(filename, np.array(frames), fps=30)


def make_jason(env, video_name, fighter_0, jammer_0, reward):
    filename = video_dir + video_name + '.json'

    results = dict()
    if (reward > 0) and (env.jammer.on < .5):
        results['Mission'] = 'Success without using Jammer'
    elif (reward > 0) and (env.jammer.on > .5):
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
    env_name = 'myenv-v2'
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

    for idx in range(NUM_TRIALS):
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
                make_jason(env, video_name, fighter_0, jammer_0, reward)
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_mode", type=str, default="human", help="rgb, if training")
    args = parser.parse_args()

    main()
