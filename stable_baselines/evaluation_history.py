import numpy as np
import matplotlib.pyplot as plt

ALPHA = 0.9


def compute_ema(results_mean, alpha):
    ema_list = []

    ema = results_mean[0][0]
    ema_list.append(ema)

    for t in range(len(results_mean[:]) - 1):
        ema = alpha * ema + (1 - alpha) * results_mean[t][0]
        ema_list.append(ema)

    return ema_list


def plot_results(timesteps, results_mean, ema_list, alpha, label_name, color):
    plt.plot(timesteps, results_mean, label=label_name, color=color, alpha=0.2)
    # plt.plot(timesteps, ema_list, label=f'{label_name} ema', color=color, linestyle="dashed")
    plt.plot(timesteps, ema_list, color=color, linestyle="solid", alpha=1)


def main(color, num_env):
    """ Define ema """
    alpha = ALPHA

    # Load data of evaluation
    data_name = DATA_NAME
    data = np.load(data_name)
    # print(data.files)

    timesteps = data['timesteps']
    results = data['results']
    sample_num = results.shape[1]

    """ Compute mean of results over samples """
    results_mean = np.mean(results, axis=1)

    """ Compute exponential moving average of results_mean """
    ema_list = compute_ema(results_mean, alpha)

    """ Plot results """
    label_name = LABEL_NAME
    plot_results(timesteps / num_env, results_mean, ema_list, alpha, label_name, color)

    return sample_num


if __name__ == '__main__':
    colorlist = ["b", "g", "r", "m", "c", "y", "k"]
    num_env_lsit = [1, 4, 8, 16, 32]

    """
    for i, num_env in enumerate(num_env_lsit):
        i = i % len(num_env_lsit)
        LABEL_NAME = 'ppo2_' + str(num_env)
        DATA_NAME = './logs/myenv-v1-ppo2/' + LABEL_NAME + '/evaluations.npz'
        sample_num = main(colorlist[i], num_env)
    """

    LABEL_NAME = 'ppo2_8'
    DATA_NAME = './logs/myenv-v2-ppo2/' + LABEL_NAME + '/evaluations.npz'
    sample_num = main(colorlist[2], num_env=8)

    """
    LABEL_NAME = 'a2c_8'
    DATA_NAME = './logs/myenv-v0-a2c/' + LABEL_NAME + '/evaluations.npz'
    sample_num = main(colorlist[6], num_env=8)

    LABEL_NAME = 'dqn'
    DATA_NAME = './logs/myenv-v0-dqn/' + LABEL_NAME + '_1/evaluations.npz'
    sample_num = main(colorlist[0], num_env=1)
    """

    plt.title(f'Average of returns & EMA(alpha={ALPHA}) over {sample_num} episodes')
    # plt.gcf().text(0.15, 0.95, f'data name:{data_name}')
    plt.xlabel('steps / num_env')
    plt.ylabel(f'Average of returns over {sample_num} episodes')
    plt.legend()
    plt.grid()
    plt.show()
