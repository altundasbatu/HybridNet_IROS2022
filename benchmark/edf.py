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
from env.mrc_problem import MRCProblem
from env.hybrid_team import HybridTeam

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='../data/small_test_set')
    parser.add_argument('--noise', type=str, default="False")
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=200)
    
    args = parser.parse_args()

    noise = False
    if args.noise == 'True' or args.noise == 'true':
        noise = True

    print('Baseline: earlist deadline first')
    save_name = './johnsonU/r2t100_001_edf_v1_2'
    folder = args.folder
    start_no = args.start
    end_no = args.end
    total_no = end_no - start_no + 1
    infeasible_coefficient = 1.0
    repeat = args.repeat

    print(folder)

    feas_count = [0 for i in range(repeat)]
    efficiency_metric = [[] for i in range(repeat)]
    results = [[] for i in range(repeat)]
    record_time = [[] for i in range(repeat)]
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
                                infeasible_makespan[i].append(env.problem.max_deadline)
                                break
                            elif env.partialw.shape[0]==(env.problem.num_tasks+1):
                                task_count[i] += env.problem.num_tasks
                                feas_count[i] += 1
                                dqn_opt = env.min_makespan
                                print('Feasible solution found, min makespan: %f' 
                                    % (env.min_makespan))
                                results[i].append([graph_no, dqn_opt])
                                makespan[i].append(env.min_makespan)
                                infeasible_makespan[i].append(env.min_makespan)
                                terminate = True
                                break
            
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
                
            end_t = time.time()
            total_time = start_t - end_t
            record_time[i].append(total_time)
            print('Time: {:.4f} s'.format(total_time))    
            print('Num feasible:', feas_count[i])

    record_time_np = -1 * np.array(record_time, dtype=np.float32)
    # print(record_time_np)
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
    feas_percentage = 100 * feas_count_np / (total_no)
    print("Feasible Percentage: {}, stdev: {}%".format(np.mean(feas_percentage), np.std(feas_percentage)))