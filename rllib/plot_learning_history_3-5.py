import numpy as np
import matplotlib.pyplot as plt

LEN = 35

def main():
    file_name = 'trial_' + str(4)
    trial_1 = np.load('./myenv-v0-ppo/learning_history/' + file_name + '.npz')
    episode = trial_1['arr_0'][:LEN]
    success = trial_1['arr_1'][:LEN]
    plt.plot(episode, success * 100, label='RL planner')

    file_name = 'trial_' + str(3)
    trial_1 = np.load('./myenv-v0-ppo/learning_history/' + file_name + '.npz')
    episode = trial_1['arr_0'][:LEN]
    success = trial_1['arr_1'][:LEN]
    plt.plot(episode, success * 100, label='RL + GAN planner, plan_margin = 0.7')

    file_name = 'trial_' + str(5)
    trial_1 = np.load('./myenv-v0-ppo/learning_history/' + file_name + '.npz')
    episode = trial_1['arr_0'][:LEN]
    success = trial_1['arr_1'][:LEN]
    plt.plot(episode, success * 100, label='RL + GAN planner, plan_margin = 0.5')

    plt.legend()
    plt.title('GAN and RL combined planner training history')
    plt.xlabel('training iteration')
    plt.ylabel('success ratio [%]')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
