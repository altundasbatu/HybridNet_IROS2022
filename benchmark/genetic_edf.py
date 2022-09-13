# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:41:10 2019

@author: pheno
Baselines - earlist deadline first
    compare with gurobi quantity and quality
    v1 rank worker from lowest average to highest average
"""

import os
import sys

script_dir = os.path.abspath( os.path.dirname( __file__ ) )
print( script_dir )
sys.path.append(script_dir)

import numpy as np
import time
import argparse

# from edfutils import workerTeam, pick_task
from edfutils import pick_task

sys.path.append(script_dir + '/../')
# from utils import SchedulingEnv
from env.scheduling_env import SchedulingEnv
from env.multi_round_scheduling_env import MultiRoundSchedulingEnv
from env.mrc_problem import MRCProblem
from env.hybrid_team import HybridTeam

from evolutionary_algorithm import *

if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--folder', type=str, default='../data/small_test_set')
    parser.add_argument('--noise', type=str, default="False")
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=200)

    parser.add_argument('--generation', type=int, default=10)
    parser.add_argument('--base-population', type=int, default=90)
    parser.add_argument('--new-random', type=int, default=10)
    parser.add_argument('--new-mutation', type=int, default=10)

    # python soft_edf.py --folder=../tmp/small_test_set --repeat=10 --base-population=90 --new-random=10 --new-mutation=10 --generation=10 --start=1 --end=200
    args = parser.parse_args()
    
    # random seed
    seed = args.seed
    random.seed(seed)
    
    noise = False
    if args.noise == 'True' or args.noise == 'true':
        noise = True

    print('Baseline: Soft Earlist Deadline First')
    save_name = './johnsonU/r2t100_001_edf_v1_2'
    folder = args.folder
    start_no = args.start
    end_no = args.end
    total_no = end_no - start_no + 1
    infeasible_coefficient = 1.0
    repeat = args.repeat

    # Evolutionary Optimization:
    generation = args.generation
    base_population = args.base_population
    new_random = args.new_random
    new_mutation = args.new_mutation
            
    print(folder)

    efficiency_metric = [[] for i in range(repeat)]
    results = [[] for i in range(repeat)]
    record_time = [[] for i in range(repeat)]

    feas_count = [0 for i in range(repeat)]
    makespan = [[] for i in range(repeat)]
    infeasible_makespan = [[] for i in range(repeat)]
    task_count = [0 for i in range(repeat)]
    total_tasks = [0 for i in range(repeat)]
    for i in range(repeat):
        for graph_no in range(start_no, end_no+1):
            print('Evaluation on {}/{}'.format(graph_no, total_no))
            start_t = time.time()
            
            fname = folder + '/problem_%04d' % graph_no
            env = SchedulingEnv(fname, restrict_same_time_scheduling=True, infeasible_coefficient=infeasible_coefficient, noise = noise)
            total_tasks[i] += env.problem.num_tasks
            workers = env.team
            terminate = False
            total_reward = 0
            schedule = []
            for t in range(int(env.problem.max_deadline)):
                exclude = []
                worker_chosen = workers.pick_worker_by_min_dur(t, env, 'v1', exclude)
                # Repeatedly select worker with minimum duration until none is available
                while worker_chosen is not None:
                    valid_task = env.get_valid_tasks(t)
                    if len(valid_task) > 0:
                        # get an updated version of STN w.r.t the chosen worker and valid tasks
                        min_rSTN, consistent = env.get_rSTN(worker_chosen, valid_task)
                        if consistent:
                            task_chosen = pick_task(min_rSTN, valid_task, t)
                        else:
                            task_chosen = -1
                        if task_chosen >= 0:
                            task_dur = env.dur[task_chosen-1][worker_chosen]
                            rt, reward, done, _ = env.step(task_chosen, worker_chosen, 1.0)
                            workers.update_status(task_chosen, worker_chosen, task_dur, t)
                            total_reward += reward
                            #print(task_chosen, worker_chosen, task_dur)
                            '''
                            Check for termination
                            '''
                            if rt == False:
                                print('Infeasible after %d insertions' % (len(env.partialw)-1))
                                task_count[i] += (len(env.partialw)-1)
                                results[i].append([graph_no, -1])
                                terminate = True
                                # 1.0 - env.problem.max_deadline/env.problem.max_deadline = 0
                                # efficiency_metric[i].append(0)
                                # infeasible_makespan[i].append(env.problem.max_deadline)
                                # makespan = env.problem.max_deadline
                                break
                            elif env.partialw.shape[0]==(env.problem.num_tasks+1):
                                schedule.append([task_chosen, worker_chosen, 1.0])
                                task_count[i] += env.problem.num_tasks
                                # Add the step to the schedule
                                feasible = True
                                dqn_opt = env.min_makespan
                                print('Feasible solution found, min makespan: %f' 
                                    % (env.min_makespan))
                                results[i].append([graph_no, dqn_opt])
                                # best_makespan = env.min_makespan
                                # makespan[i].append(env.min_makespan)
                                # infeasible_makespan[i].append(env.min_makespan)
                                terminate = True
                                # e_metric = 1.0 - (env.min_makespan / env.problem.max_deadline)
                                # efficiency_metric[i].append(e_metric)
                                break
                            schedule.append([task_chosen, worker_chosen, 1.0])
        
                            # Attempt to pick another worker
                            worker_chosen = workers.pick_worker_by_min_dur(t, env, 'v1', exclude)
                        else:
                            # No valid tasks for this worker, move to next
                            exclude.append(worker_chosen)
                            worker_chosen = workers.pick_worker_by_min_dur(t, env, 'v1', exclude)
                    else:
                        break
                if terminate:
                    break
                
            # if there are unscheduled_tasks, add them to the schedule with worst worker in total
            unscheduled_tasks = env.get_unscheduled_tasks()
            # print(schedule, env.min_makespan)
            # print("Unscheduled:", unscheduled_tasks)
            # if infeasible, complete the schedule
            if len(unscheduled_tasks) > 0:
                worker = env.problem.get_worst_worker(unscheduled_tasks-1)
                for u_task in unscheduled_tasks:
                    schedule.append([u_task, worker, 1.0])
            # print(schedule, env.min_makespan)
            # Run Evolutionary Algorithm to generate more schedules:
            env.reset() # Reset the Single Round Environment
            new_random_schedules = random_gen_schedules([schedule], env.team, base_population + new_random - 1)
            if len(new_random_schedules) == 0: # if there can be no swaps made
                new_random_schedules = generate_evolution([schedule], base_population + new_random - 1, [0])
            new_mutations = generate_evolution([schedule], new_mutation, [0])
            new_generation = [schedule] + new_random_schedules + new_mutations # include the baseline
            # Store the previously generated scores
            scores = [[], []] # Scores for Infeasible and Feasible Schedule Scores
            schedules = [[], []] # Schedules for Infeasible and Feasible Schedule Scores
            # multi_round_env = MultiRoundSchedulingEnv(env.problem, env.team)
            infeasible_idx_sorted = []
            feasible_idx_sorted = []
            for gen in range(generation):
                # print(scores[1])
                # indices = [[], []] # Indicies for Infeasible and Feasible Schedules
                # Run MultiRound to get total score/feasibility for the new generation
                for j in range(len(new_generation)):
                    env.reset()
                    # print(schedule_i)
                    schedule_i = new_generation[j]
                    # print(schedule_i)
                    rt = False
                    for step in schedule_i:
                        # print(step)
                        rt, reward, done, _ = env.step(step[0], step[1], step[2])
                        if rt == False: # Infeasible
                            scores[0].append(env.problem.max_deadline)
                            schedules[0].append(schedule_i)
                            break
                    if rt:
                        scores[1].append(env.min_makespan)
                        schedules[1].append(schedule_i)
                    # if j == 0: print(scores)
                # Select the top evolution_cutoff indices for next generation
                # print(indices, scores)
                if len(scores[0]) != 0:
                    infeasible_idx_sorted = np.argsort(scores[0])
                if len(scores[1]) != 0:
                    feasible_idx_sorted = np.argsort(scores[1])
                # print(infeasible_idx_sorted, feasible_idx_sorted)                
                
                # if gen < generation - 1: # For all but last step, take the top base_population of the scores and schedules
                feasible_top_n = min(base_population, len(scores[1]))
                infeasible_top_n = max(0, base_population - len(scores[1]))
                # Get the top elements of the array
                schedules[1] = np.array(schedules[1], dtype=int)[feasible_idx_sorted][:feasible_top_n].tolist() 
                scores[1] = np.array(scores[1], dtype=int)[feasible_idx_sorted][:feasible_top_n].tolist()
                if infeasible_top_n > 0:
                    schedules[0] = np.array(schedules[0], dtype=int)[infeasible_idx_sorted][:infeasible_top_n].tolist()
                    scores[0] = np.array(scores[0], dtype=int)[infeasible_idx_sorted][:infeasible_top_n].tolist()
                else:
                    schedules[0] = []
                    scores[0] = []
                # print(schedules[0] + schedules[1])
                # Generate new_random number of random schedules
                random_schedules = random_gen_schedules(schedules[0] + schedules[1], env.team, new_random)
                if len(random_schedules) == 0:
                    random_schedules = swap_task_allocation(schedules[0] + schedules[1], new_random)
                # Generate new_mutation number of mutation schedules
                new_mutation_schedules = swap_task_allocation(schedules[0] + schedules[1], new_mutation)
                if len(new_mutation_schedules) == 0:
                    new_mutation_schedules = random_gen_schedules(schedules[0] + schedules[1], env.team, new_mutation)
                new_generation = random_schedules + new_mutation_schedules
                # update schedules to baseline
                # schedules = baselines
                    
            if len(feasible_idx_sorted) != 0: # there is a feasible solution
                idx = 0 # feasible_idx_sorted[0]
                # print(idx, len(scores[1]))
                print(scores[1])
                print(scores[1][idx])
                # print(makespan[i])
                makespan[i].append(scores[1][idx])
                infeasible_makespan[i].append(scores[1][idx])
                feas_count[i] += 1
            else: # the solution is infeasible
                idx = 0 # infeasible_idx_sorted[0]
                # infeasible_makespan[i].append(scores[0][idx])
                # give worst case:
                # env.reset()
                infeasible_makespan[i].append(env.problem.max_deadline)
                
            end_t = time.time()
            time_t = end_t - start_t
            record_time[i].append(time_t)
            print('Time: {:.4f} s'.format(time_t))    
            print('Num feasible:', feas_count)

        # # save results
        # results_np = np.array(results[i], dtype=np.float32)
        # np.save(save_name, results_np)
        
        # save computation time
        # record_time_np = np.array(record_time[i], dtype=np.float32)
        # np.save(save_name+'_time_'+str(i)+'.txt', record_time_np)

    record_time_np = np.array(record_time, dtype=np.float32)
    print(record_time_np)
    print("Tasks:", task_count, total_tasks)
    # result summary
    feas_count_np = np.array(feas_count)
    print('Feasible solution found: {}/{}'.format(np.sum(feas_count), total_no*repeat))
    print('Average time per instance:  {:.4f}, stdev: {:.4f}'.format(np.mean(record_time_np), np.std(record_time_np)))
    # print(makespan)
    feasible_makespans = [np.mean(np.asarray(m)) for m in makespan]
    print("Feasible Makespan: ", np.mean(feasible_makespans), ", stdev: ", np.std(feasible_makespans))
    # print(feasible_makespans)
    infeasible_makespans = [np.mean(np.asarray(im)) for im in infeasible_makespan]
    print("Total Makespan: ", np.mean(infeasible_makespans), ", stdev: ", np.std(infeasible_makespans))
    feas_percentage = 100 * feas_count_np / total_no
    print("Feasible Percentage: {}, stdev: {}%".format(np.mean(feas_percentage), np.std(feas_percentage)))
