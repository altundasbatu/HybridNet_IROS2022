# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 20:27:51 2020

@author: baltundas3

Evaluate in a loop / plot saved in a folder without showing
"""

import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

import copy

import time

import torch

from scheduler import PGScheduler
from env.mrc_problem import MRCProblem
from env.hybrid_team import HybridTeam

from env.scheduling_env import SchedulingEnv
from env.multi_round_scheduling_env import MultiRoundSchedulingEnv

def run_test(fname, save_folder_prob, scheduler, num_rounds, batch_size, mode, infeasible_coefficient, noise = False, with_est = True, est_noise = True, genetic=False):
    human_learning = True
    raw_makespan = []
    net_makespan = []
    feasible_solution_count = 0
    
    # load env from data folder
    problem = MRCProblem(fname = fname, max_deadline_multiplier=infeasible_coefficient, noise = noise)

    # Create a Team
    team = HybridTeam(problem)
                
    # scheduler already being loaded outside of this function    
    if mode == 'sample':
        # Create multiple instances of the same environment, since the environments update after every round
        multi_round_envs = [MultiRoundSchedulingEnv(problem, team) for i in range(batch_size)]
        for i_b in range(batch_size):
            print('Batch {}/{}.'.format(i_b+1, batch_size))
            for step_count in range(num_rounds):
                schedule = scheduler.select_action(multi_round_envs[i_b].get_single_round(), genetic)
                success, reward, done, makespan = multi_round_envs[i_b].step(schedule, human_learning=human_learning)
                if success: # the generated schedule is feasible
                    feasible_solution_count += 1
                    net_makespan.append(makespan)
                    raw_makespan.append(makespan)
                else: # infeasible schedule generated
                    raw_makespan.append(multi_round_envs[i_b].problem.max_deadline)
                print('Makespan: {:.4f}'.format(makespan)) #, end='\r')
    elif mode == 'argmax':
        multi_round_env = MultiRoundSchedulingEnv(problem, team)
        for step_count in range(num_rounds):
            schedule = scheduler.select_action(multi_round_env.get_single_round(), genetic)
            success, reward, done, makespan = multi_round_env.step(schedule, human_learning=human_learning)
            if success: # the generated schedule is feasible
                feasible_solution_count += 1
                net_makespan.append(makespan) # record reward if it is feasible
                raw_makespan.append(makespan)
            else: # infeasible schedule generated
                raw_makespan.append(multi_round_env.problem.max_deadline)
            print('Makespan: {:.4f}'.format(makespan)) # , end='\r')
    elif mode == 'best':
        # Create multiple instances of the same environment, since the environments update after every round
        multi_round_envs = [MultiRoundSchedulingEnv(problem, team) for i in range(batch_size)]
        actual_env = MultiRoundSchedulingEnv(problem, team)
        for step_count in range(num_rounds):
            schedules = []
            success_list = []
            batch_net_makespans = []
            batch_all_makespans = []
            # Generate Multiple Schedules for Ensemble Schedule Generation
            for i_b in range(batch_size):
                # print('Batch {}/{}.'.format(i_b+1, batch_size))
                env = None
                if with_est:
                    env = multi_round_envs[i_b].get_estimate_environment(est_noise=est_noise)
                else:
                    env = multi_round_envs[i_b].get_actual_environment(human_noise=noise)
                schedule = scheduler.select_action(env, genetic=genetic)
                schedules.append(schedule)
                success, reward, done, makespan = multi_round_envs[i_b].step(schedule, human_learning=human_learning, evaluate=with_est, human_noise=noise, estimator_noise=est_noise)
                success_list.append(success)
                if success: # the generated schedule is feasible
                    batch_net_makespans.append(makespan)
                    batch_all_makespans.append(makespan)
                else: # infeasible schedule generated
                    batch_all_makespans.append(multi_round_envs[i_b].problem.max_deadline)
                print('Makespan: {:.4f}'.format(makespan), end='\r')
            # select the batch that is the best
            # Select the Feasible Smallest Makespan
            batch_success_np = np.array(success_list)
            batch_makespans_np = np.array(batch_all_makespans)
            # batch_rewards_np = np.array(batch_rewards)
            idx = np.argmin(batch_makespans_np) # All are infeasible
            if np.any(batch_success_np):
                # If there is a feasible schedule, use the one with minimum makespan
                idx = np.argwhere(batch_success_np)[np.argmin(batch_makespans_np[batch_success_np])][0]
            # Select the Schedule with expected best performance
            actual_schedule = schedules[idx]
            a_success, a_reward, a_done, a_makespan = actual_env.step(actual_schedule, human_learning=human_learning, evaluate=False, human_noise=noise)
            if a_success:
                feasible_solution_count += 1
                net_makespan.append(a_makespan)
                raw_makespan.append(a_makespan)
            else:
                raw_makespan.append(a_makespan)
            
            # Update Emulated Environment based on the Emulated Best Schedule Environment
            best_env = multi_round_envs[idx]
            multi_round_envs = [copy.deepcopy(best_env) for i in range(batch_size)]
            
    return net_makespan, raw_makespan, feasible_solution_count

def get_average_max_makespan(data_folder, start_no=1, end_no=200):
    makespan = []
    for prob_no in range(start_no, end_no+1):
            fname = data_folder + '/problem_' + format(prob_no, '04')
            problem = MRCProblem(fname = fname)
            makespan.append(problem.max_deadline)
    return np.array(makespan).mean()

if __name__ == '__main__':
    """
    python pg_eval medium_training_set_checkpoint_02000.tar  
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--data-folder', type=str, default="tmp/small_test_set")
    parser.add_argument('--save-folder', type=str, default="tmp/small_test_results")
    parser.add_argument('--infeasible-coefficient', type=float, default=1.0)
    parser.add_argument('--human-noise', action='store_true')
    parser.set_defaults(human_noise=False)
    parser.add_argument('--estimator', action='store_true')
    parser.set_defaults(estimator=False)
    parser.add_argument('--estimator-noise', action='store_true')
    parser.set_defaults(estimator_noise=False)
    
    parser.add_argument('--genetic', dest='genetic', action='store_true')
    parser.add_argument('--no-genetic', dest='genetic', action='store_false')
    parser.set_defaults(genetic=False)
    # Test Information
    parser.add_argument('--start-no', default=1, type=int)
    parser.add_argument('--end-no', default=200, type=int)
    # Checkpoint Selection
    parser.add_argument('--cp', type=str, default="tmp/small_training_set/checkpoints_20_pg")
    parser.add_argument('--specific-cp', type=str, default=None)
    parser.add_argument('--start-cp', default=8000, type=int)
    parser.add_argument('--end-cp', default=8000, type=int)
    parser.add_argument('--cp-period', default=400, type=int)
    # Batch Count
    parser.add_argument('--batch-size', default=8, type=int)
    # Round Count
    parser.add_argument('--num-rounds', default=4, type=int)
    parser.add_argument('--mode', default='argmax', type=str)
    parser.add_argument('--nn', default='hybridnet', type=str)
    parser.add_argument('--repeat', default=1, type=int)
    
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--device-id', default=0, type=int)
    parser.add_argument('--verbose', default='none', type=str)
    
    args = parser.parse_args()
    
    # random seed
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)

    data_folder = args.data_folder
    # print(get_average_max_makespan(data_folder))
    
    save_folder = args.save_folder
    
    infeasible_coefficient = args.infeasible_coefficient
    
    real_noise = args.human_noise
    estimator = args.estimator
    est_noise = args.estimator_noise
    genetic = args.genetic
    print("Genetic Eval:", genetic)
    start_no = args.start_no
    end_no = args.end_no
    total_no = end_no - start_no + 1
    batch_size = args.batch_size
    num_rounds = args.num_rounds
    
    mode = args.mode
    nn = args.nn
    repeat = args.repeat
    verbose = args.verbose
    
    cp_parent = args.cp
    

    net_mean = []
    std_net = []
    raw_mean = []
    std_raw = []
    feasibility_counts = []
    times = []

    # Files:    
    # Iterating over give periods through addition is easier as it allows for more direct control of start and end checkpoints
    start_checkpoint = args.start_cp
    end_checkpoint = args.end_cp # Included
    checkpoint_period = args.cp_period
    # Iterate:
    checkpoint = start_checkpoint
    while checkpoint <= end_checkpoint:
        net_makespan_list = []
        raw_makespan_list = []
        print("Evaluating: ", checkpoint)
        checkpoint_folder = cp_parent + "/checkpoint_%05d.tar" % checkpoint
        # if an exact cp is provided, set up so that the evaluation ends after running
        if args.specific_cp is not None:
            checkpoint_folder = args.specific_cp
            end_checkpoint = checkpoint
        '''
        Load model
        '''
        # scheduler = PGScheduler(device=torch.device('cpu',0))
        scheduler_mode = mode
        if mode == 'best':
            scheduler_mode = 'sample'
        
        scheduler = PGScheduler(device=torch.device(args.device, args.device_id), nn=nn, selection_mode=scheduler_mode, verbose=verbose)
        scheduler.load_checkpoint(checkpoint_folder)
        print('Loaded: '+checkpoint_folder)
        
        print('Evaluation starts.')
        print('Save Folder: '+ save_folder)

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        
        for count in range(repeat):
            feasibility_count = 0
            for prob_no in range(start_no, end_no+1):
                fname = data_folder + '/problem_' + format(prob_no, '04')
                
                save_folder_prob = save_folder + '/%05d' % prob_no
                # if not os.path.exists(save_folder_prob):
                #     os.makedirs(save_folder_prob)

                print('Evaluation {} on {}/{}.'.format(count + 1, prob_no-start_no + 1, total_no))
                start = time.time()
                net_makespan, raw_makespan, feasibility_of_problem = run_test(fname, save_folder_prob, scheduler, num_rounds, batch_size, mode, infeasible_coefficient, real_noise, estimator, est_noise, genetic)
                end= time.time()
                times.append(start - end)
                feasibility_count += feasibility_of_problem
                net_makespan_list.extend(net_makespan)
                raw_makespan_list.extend(raw_makespan)
                # print(rewards, feasibility_count)
            net_makespan_array = np.array(net_makespan_list)
            raw_makespan_array = np.array(raw_makespan_list)
            feasibility_counts.append(feasibility_count)
            # reward_folder = "tmp/small_training_set/eval_rewards_%05d.txt" % checkpoint
            # print(rewards_array, feasibility_count)
            # print('Mean: {}, Std: {}'.format(np.mean(raw_makespan_array), np.std(raw_makespan_array)))
            net_mean.append(np.mean(net_makespan_array))
            std_net.append(np.std(net_makespan_array))
            raw_mean.append(np.mean(raw_makespan_array))
            std_raw.append(np.std(raw_makespan_array))
            # print(net_mean, std_net, raw_mean, std_raw, feasibility_counts)    
        checkpoint += checkpoint_period
        
    num_tests = (end_no - start_no + 1)
    # print(feasibility_count, num_rounds, num_tests)
    feasibility_percentages = 100 * np.array(feasibility_counts) / (num_rounds * num_tests)
    single_instance_times = -1 * np.array(times) / num_rounds
    print("Total Makespan, Total Makespan Stdev, Feasibility, Feasibility Stdev, Time, Time_Stdev")
    print(np.mean(np.array(raw_mean)), np.std(np.array(raw_mean)), np.mean(np.array(feasibility_percentages)), np.std(feasibility_percentages), np.mean(single_instance_times), np.std(single_instance_times))
    print('Done.')