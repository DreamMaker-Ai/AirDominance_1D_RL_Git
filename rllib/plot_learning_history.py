import numpy as np
import matplotlib.pyplot as plt


def main():
    plt_list = [6,7]

    for i in plt_list:
        file_name = 'trial_' + str(i)
        trial_1 = np.load('./myenv-v0-ppo/learning_history/' + file_name + '.npz')
        episode = trial_1['arr_0']
        success = trial_1['arr_1']
        plt.plot(episode, success * 100, label=file_name)

    plt.legend()
    plt.title('GAN and RL combined planner training history')
    plt.xlabel('training iteration')
    plt.ylabel('average success ratio [%] over 1000 episodes')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
