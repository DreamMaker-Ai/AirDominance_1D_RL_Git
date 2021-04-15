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


def main(color, num_env, start_step=0):
    """ Define ema """
    alpha = ALPHA

    # Load data of evaluation
    data_name = DATA_NAME
    data = np.load(data_name)
    # print(data.files)

    timesteps = data['timesteps'] + start_step
    results = data['results']
    sample_num = results.shape[1]

    """ Compute mean of results over samples """
    results_mean = np.mean(results, axis=1)

    """ Compute exponential moving average of results_mean """
    ema_list = compute_ema(results_mean, alpha)

    """ Plot results """
    label_name = LABEL_NAME
    plot_results(timesteps / num_env, results_mean, ema_list, alpha, label_name, color)

    dt = timesteps[-1] - timesteps[-2]
    tf = timesteps[-1]

    return sample_num, tf, dt


if __name__ == '__main__':
    LABEL_NAME = 'ppo2_8'
    DATA_NAME = './logs/myenv-v1-ppo2/' + LABEL_NAME + '/evaluations_0.npz'
    sample_num, tf, dt = main("m", num_env=8)

    start_step = tf + dt

    DATA_NAME = './logs/myenv-v1-ppo2/' + LABEL_NAME + '/evaluations.npz'
    sample_num, tf, dt = main("r", num_env=8, start_step=start_step)

    plt.title(f'Average of returns & EMA(alpha={ALPHA}) over {sample_num} episodes')
    plt.gcf().text(0.25, 0.95, f'Data:{DATA_NAME}')
    plt.xlabel('steps / num_env')
    plt.ylabel(f'Average of returns over {sample_num} episodes')
    # plt.legend()
    plt.grid()
    plt.show()
