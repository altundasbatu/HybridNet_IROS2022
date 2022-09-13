

import numpy as np
import matplotlib.pyplot as plt

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--folder', type=str, default="tmp/small_training_set")
    parser.add_argument('--checkpoint', type=str, default="tmp/small_training_set/checkpoints_21_pg")
    parser.add_argument('--count', type=int, default=2000)
    parser.add_argument('--mode', type=str, default='pg')
    
    args = parser.parse_args()
    
    # folder = args.folder
    cp_path = args.checkpoint
    path = cp_path + '/feasible_solution_count.txt'

    count = args.count
    num_schedules = 32
    if args.mode == 'gb':
        num_schedules = 4
    
    # path = "tmp/small_test_results/checkpoints_33_pg_eval_count_feasibility.txt"
    print(path)
    feasible_count = np.loadtxt(path)
    length = len(feasible_count)
    epochs = range(1, len(feasible_count)+1)
    print(length)

    # # Moving Average
    # plt.plot(np.convolve(feasible_count, np.ones(count)/count, mode='full')[count:-count])
    data = []
    ep = []
    for i in range(int(length/count)):
        # print(i * count, (i+1) * count)
        data.append(np.mean(feasible_count[i * count: (i + 1) * count]))
        ep.append(i)
    print(data)
    plt.plot(data)
    
    plt.ylabel("Local Average of Feasibility")
    plt.xlabel("Training Epochs")
    plt.show()
    plt.clf()

    epochs = range(1, len(feasible_count)+1)
    # Moving Average
    data = np.convolve(feasible_count, np.ones(count)/count, mode='full')[count:-count] / num_schedules * 100
    print(data)
    plt.plot(data)
    plt.ylabel("Local Average of Feasibility (%)")
    plt.xlabel("Training Epochs")
    plt.show()
    plt.clf()
    
    
    # folder = 'tmp/small_training_set'
    path = cp_path + '/efficiency_metrics.txt'

    # path = "tmp/small_test_results/checkpoints_33_pg_eval_mean_feasible.txt"
    efficiency = np.loadtxt(path)
    plt.plot(feasible_count[-3*count:])
    plt.show()
    plt.clf()
    epochs = range(1, len(efficiency)+1)
    # Moving Average
    data = np.convolve(efficiency, np.ones(count)/count, mode='full')[count:-count]
    plt.plot(data)
    # with np.printoptions(threshold=np.inf):
    print(data)
    plt.ylabel("Running Average of Efficiency")
    plt.xlabel("Training Epochs")
    plt.show()
    plt.clf()
    
    # plt.ylabel("Mean Feasible Schedule out of 32")
    # plt.show()


    # feasibility_count_file = "tmp/small_test_results/checkpoints_13_pg_eval_feasibility.txt"
    # feasibility_training = np.loadtxt(feasibility_count_file)
    
    # mean_rewards_file = "tmp/small_test_results/cp_15_net.txt"
    # mean_rewards_raw = np.loadtxt(mean_rewards_file)
    # print(mean_rewards_raw.shape)
    # mean_rewards = mean_rewards_raw[:, 0]
    # stdt = mean_rewards_raw[:, 1]
    # print(mean_rewards)
    
    # epochs_2 = np.array(range(1, len(mean_rewards))) * 50
    # epochs_2 = np.insert(epochs_2, 0, 10)
    # plt.plot(epochs_2, mean_rewards)
    # # plt.fill_between(epochs_2, mean_rewards-stdt, mean_rewards+stdt ,alpha=0.3)
    
    # plt.xlabel("Training Epochs")
    # plt.ylabel("Mean Greedy Makespan for 200 Test Set")
    # plt.show()
    # plt.xlabel("Training Epochs")
    # plt.ylabel("Total number of Feasible Solutions for 200 Test Set (max. 6400)")
    # plt.plot(epochs_2, feasibility_training)
    # plt.show()
    
    
