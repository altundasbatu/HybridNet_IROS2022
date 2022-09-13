# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:46:16 2020

@author: pheno

Evaluation code using HetNet to select tasks
v0 workers are being picked in their idx order
v1 rank worker from lowest average to highest average
"""

import argparse
import sys
import time
import os
import copy

import numpy as np
import torch


sys.path.append('../')

from benchmark import benchmark_utils
from env.hybrid_team import HybridTeam
from env.scheduling_env import SchedulingEnv
from hetnet import HybridScheduleNet, HybridScheduleNet4Layer
from utils import hetgraph_node_helper, build_hetgraph


in_dim = {'task': 6, # do not change
        #   'loc': 1,
          'worker': 1,
          'state': 4
          }

hid_dim = {'task': 64,
        #    'loc': 64,
           'worker': 64,
           'state': 64
           }

out_dim = {'task': 32,
        #   'loc': 32,
          'worker': 32,
          'state': 32
          }

cetypes = [('task', 'temporal', 'task'),
        #    ('task', 'located_in', 'loc'),('loc', 'near', 'loc'),
           ('task', 'assigned_to', 'worker'), ('worker', 'com', 'worker'),
           ('task', 'tin', 'state'), # ('loc', 'lin', 'state'), 
           ('worker', 'win', 'state'), ('state', 'sin', 'state'), 
           ('task', 'take_time', 'worker'), ('worker', 'use_time', 'task')]

num_heads = 8

# TODO: Ignore Map Width Mapping
num_worker_to_map_width = {
    2: 2,
    3: 2,
    5: 3,
    10: 5
}

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--cpu', default=False, action='store_true')
parser.add_argument('--data-path', default='../tmp/test1', type=str)
parser.add_argument('--start-no', default=1, type=int)
parser.add_argument('--end-no', default=10, type=int)
parser.add_argument('--version', default='v3', type=str)
parser.add_argument('--suffix', type=str)
args = parser.parse_args()

assert args.version in ['v0', 'v1', 'v2', 'v3']

device = torch.device('cpu' if args.cpu else 'cuda')

policy_net = HybridScheduleNet(in_dim, hid_dim, out_dim, cetypes).to(device)
# policy_net = HybridScheduleNet4Layer(in_dim, hid_dim, out_dim, cetypes, num_heads).to(device)
policy_net.eval()

if args.checkpoint is not None:
    print('Evaluating: ' + args.checkpoint)
    cp = torch.load(args.checkpoint, map_location=device)
    policy_net.load_state_dict(cp['policy_net_state_dict'])
    training_steps_done = cp['training_steps']
    print('Model status: trained with %d steps' % training_steps_done)

    checkpoint_id = args.checkpoint.split('/')[-1].replace('.tar', '')
else:
    checkpoint_id = ''

print('Method: earlist deadline first / GNN version')

folder = args.data_path
last_folder_name = os.path.basename(os.path.normpath(folder))
print(last_folder_name, 'at', folder)
save_name = benchmark_utils.get_save_name(args.version, last_folder_name, checkpoint_id)
print('Saving to', save_name)
start_no = args.start_no
end_no = args.end_no
total_no = end_no - start_no + 1
feas_count = 0

results = []
record_time = []

results_folder = benchmark_utils.get_results_folder_name(last_folder_name, args.start_no, args.end_no,
                                                         args.suffix)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

result_save_path = os.path.join(results_folder, save_name)
time_save_path = os.path.join(results_folder, save_name + '_time')
print(f'Saving results to {result_save_path}, saving time to {time_save_path}')

for graph_no in range(start_no, end_no+1):
    print('Evaluation on {}/{}'.format(graph_no, total_no))
    start_t = time.time()

    fname = folder + '/%05d' % graph_no
    print("Environment save location", fname)
    env = SchedulingEnv(fname)
    map_width = num_worker_to_map_width[len(env.team)]
    
    workers = copy.deepcopy(env.team)

    terminate = False
    
    # TODO: Change * 100 to max_duration
    for t in range(env.problem.num_tasks * 100):
        ####################################################
        #                        v0                        #
        ####################################################
        if args.version == 'v0':
            worker_available = workers.pick_worker(t)

            for worker_chosen in worker_available:
                unsch_tasks = np.array(env.get_unscheduled_tasks(), dtype=np.int64)
                valid_tasks = np.array(env.get_valid_tasks(t), dtype=np.int64)

                if len(valid_tasks) > 0:
                    g = build_hetgraph(env.halfDG, env.problem.num_tasks, env.num_workers, env.dur,
                                    #    map_width, # np.array(env.loc, dtype=np.int64),
                                    #    1.0, 
                                       env.partials, unsch_tasks)
                    # g = g.to(device)

                    feat_dict = hetgraph_node_helper(env.halfDG.number_of_nodes(),
                                                     env.partialw,
                                                     env.partials, # env.loc,
                                                     env.dur,
                                                     env.num_workers,
                                                     len(valid_tasks))

                    feat_dict_tensor = {}
                    for key in feat_dict:
                        feat_dict_tensor[key] = torch.Tensor(feat_dict[key]).to(device)
                    
                    print("Valid Tasks:", valid_tasks)
                    task_chosen = gnn_pick_task(g, valid_tasks, policy_net,
                                                feat_dict_tensor)
                else:
                    task_chosen = -1

                if task_chosen >= 0:
                    task_dur = env.dur[task_chosen - 1][worker_chosen]
                    rt, reward, done, info = env.step(task_chosen, worker_chosen)
                    workers.update_status(task_chosen, worker_chosen, task_dur, t)
                    # print(task_chosen, worker_chosen, task_dur)
                    '''
                    Check for termination
                    '''
                    if rt == False:
                        print('Infeasible after %d insertions' % (len(env.partialw) - 1))
                        results.append([graph_no, -1])
                        terminate = True
                        break
                    elif env.partialw.shape[0] == (env.problem.num_tasks + 1):
                        feas_count += 1
                        dqn_opt = env.min_makespan
                        print('Feasible solution found, min makespan: %f'
                              % (env.min_makespan))
                        results.append([graph_no, dqn_opt])
                        terminate = True
                        break

            if terminate:
                break
        ####################################################
        #                  v1, v2, v3                      #
        ####################################################
        elif args.version in ['v1', 'v2', 'v3']:
            exclude = []
            worker_chosen = env.pick_worker_by_min_dur(t, args.version, exclude)
            # Repeatedly select worker with minimum duration until none is available
            while worker_chosen is not None:
                unsch_tasks = np.array(env.get_unscheduled_tasks(), dtype=np.int64)
                valid_tasks = np.array(env.get_valid_tasks(t), dtype=np.int64)

                if len(valid_tasks) > 0:
                    g = build_hetgraph(env.halfDG, env.problem.num_tasks, env.problem.num_robots, env.problem.num_humans, env.dur,
                                       # map_width, # np.array(env.loc, dtype=np.int64), 1.0,
                                       env.partials, unsch_tasks)
                    # g = g.to(device)
                    feat_dict = hetgraph_node_helper(env.halfDG.number_of_nodes(),
                                                     env.partialw,
                                                     env.partials, # env.loc,
                                                     env.dur,
                                                     len(env.team),
                                                     len(valid_tasks))

                    feat_dict_tensor = {}
                    for key in feat_dict:
                        feat_dict_tensor[key] = torch.Tensor(feat_dict[key]).to(device)
                    # print(valid_tasks, policy_net, feat_dict_tensor)
                    task_chosen = gnn_pick_task(g, valid_tasks, policy_net, feat_dict_tensor)

                    if task_chosen >= 0:
                        task_dur = env.dur[task_chosen-1][worker_chosen]
                        rt, reward, done, info = env.step(task_chosen, worker_chosen)
                        workers.update_status(task_chosen, worker_chosen, task_dur, t)
                        '''
                        Check for termination
                        '''
                        if not rt:
                            print('Infeasible after %d insertions' % (len(env.partialw)-1))
                            results.append([graph_no, -1])
                            terminate = True
                            break
                        elif env.partialw.shape[0] == (env.problem.num_tasks+1):
                            feas_count += 1
                            dqn_opt = env.min_makespan
                            print('Feasible solution found, min makespan: %f'
                                  % env.min_makespan)
                            results.append([graph_no, dqn_opt])
                            terminate = True
                            break

                        # Attempt to pick another worker
                        worker_chosen = env.pick_worker_by_min_dur(t, args.version, exclude)
                    else:
                        # No valid tasks for this worker, move to next
                        exclude.append(worker_chosen)
                        worker_chosen = env.pick_worker_by_min_dur(t, args.version, exclude)
                else:
                    break

            if terminate:
                break
        
    end_t = time.time()
    record_time.append([graph_no, end_t - start_t])
    print('Time: {:.4f} s'.format(end_t - start_t))
    print('Num feasible:', feas_count)

    # save results
    results_np = np.array(results, dtype=np.float32)
    np.save(result_save_path, results_np)
    
    # save computation time
    record_time_np = np.array(record_time, dtype=np.float32)
    np.save(time_save_path, record_time_np)

# result summary
print('Feasible solution found: {}/{}'.format(feas_count, total_no))
print('Average time per instance:  {:.4f}'.format(sum(record_time_np[:,1])/total_no))
