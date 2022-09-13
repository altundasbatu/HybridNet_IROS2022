# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:46:16 2020

@author: pheno

Evaluation code using HetNet to select tasks
v0 robots are being picked in their idx order
v1 rank robot from lowest average to highest average
"""

import argparse
import sys
import time
import os

import numpy as np
import torch

from edfutils import RobotTeam

sys.path.append('../')

from benchmark import benchmark_utils

from hetnet import ScheduleNet4Layer
from utils import SchedulingEnv, hetgraph_node_helper, build_hetgraph


'''
Pick a task using GNN value function
    hetg: HetGraph in DGL
    act_task: unscheduled/available tasks
    pnet: trained GNN model
    ft_dict: input feature dict
    rj: robot chosen, not needed as hetg is based on the selected robot
'''
def gnn_pick_task(hetg, act_task, pnet, ft_dict):
    length = len(act_task)
    if length == 0:
        return -1
       
    '''
    pick task using GNN
    '''
    #idx = np.argmin(tmp)
    if length == 1:
        idx = 0
    else:
        with torch.no_grad():
            result = pnet(hetg, ft_dict)
            # Lx1
            q_s_a = result['value']
            
            # get argmax on selected robot
            a_idx = q_s_a.argmax()
            idx = int(a_idx)
    
    task_chosen = act_task[idx]

    return task_chosen


in_dim = {'task': 6,
          'loc': 1,
          'robot': 1,
          'state': 4,
          'value': 1
          }

hid_dim = {'task': 64,
           'loc': 64,
           'robot': 64,
           'state': 64,
           'value': 64
           }

out_dim = {'task': 32,
          'loc': 32,
          'robot': 32,
          'state': 32,
          'value': 1
          }

cetypes = [('task', 'temporal', 'task'),
           ('task', 'located_in', 'loc'),('loc', 'near', 'loc'),
           ('task', 'assigned_to', 'robot'), ('robot', 'com', 'robot'),
           ('task', 'tin', 'state'), ('loc', 'lin', 'state'), 
           ('robot', 'rin', 'state'), ('state', 'sin', 'state'), 
           ('task', 'tto', 'value'), ('robot', 'rto', 'value'), 
           ('state', 'sto', 'value'), ('value', 'vto', 'value'),
           ('task', 'take_time', 'robot'), ('robot', 'use_time', 'task')]

num_heads = 8

num_robot_to_map_width = {
    2: 2,
    5: 3,
    10: 5
}

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--cpu', default=False, action='store_true')
parser.add_argument('--data-path')
parser.add_argument('--start-no', default=1, type=int)
parser.add_argument('--end-no', default=1000, type=int)
parser.add_argument('--version', type=str)
parser.add_argument('--suffix', type=str)
args = parser.parse_args()

assert args.version in ['v0', 'v1', 'v2', 'v3']

device = torch.device('cpu' if args.cpu else 'cuda')

policy_net = ScheduleNet4Layer(in_dim, hid_dim, out_dim, cetypes, num_heads).to(device)
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
    env = SchedulingEnv(fname)
    map_width = num_robot_to_map_width[env.num_robots]
    
    robots = RobotTeam(env.num_robots)

    terminate = False
    
    for t in range(env.num_tasks * 10):
        ####################################################
        #                        v0                        #
        ####################################################
        if args.version == 'v0':
            robot_available = robots.pick_robot(t)

            for robot_chosen in robot_available:
                unsch_tasks = np.array(env.get_unscheduled_tasks(), dtype=np.int64)
                valid_tasks = np.array(env.get_valid_tasks(t), dtype=np.int64)

                if len(valid_tasks) > 0:
                    g = build_hetgraph(env.halfDG, env.num_tasks, env.num_robots, env.dur,
                                       map_width, np.array(env.loc, dtype=np.int64),
                                       1.0, env.partials, unsch_tasks, robot_chosen,
                                       valid_tasks)
                    g = g.to(device)

                    feat_dict = hetgraph_node_helper(env.halfDG.number_of_nodes(),
                                                     env.partialw,
                                                     env.partials, env.loc, env.dur,
                                                     map_width, env.num_robots,
                                                     len(valid_tasks))

                    feat_dict_tensor = {}
                    for key in feat_dict:
                        feat_dict_tensor[key] = torch.Tensor(feat_dict[key]).to(device)

                    task_chosen = gnn_pick_task(g, valid_tasks, policy_net,
                                                feat_dict_tensor)
                else:
                    task_chosen = -1

                if task_chosen >= 0:
                    task_dur = env.dur[task_chosen - 1][robot_chosen]
                    rt, reward, done = env.insert_robot(task_chosen, robot_chosen)
                    robots.update_status(task_chosen, robot_chosen, task_dur, t)
                    # print(task_chosen, robot_chosen, task_dur)
                    '''
                    Check for termination
                    '''
                    if rt == False:
                        print('Infeasible after %d insertions' % (len(env.partialw) - 1))
                        results.append([graph_no, -1])
                        terminate = True
                        break
                    elif env.partialw.shape[0] == (env.num_tasks + 1):
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
            robot_chosen = robots.pick_robot_by_min_dur(t, env, args.version, exclude)
            # Repeatedly select robot with minimum duration until none is available
            while robot_chosen is not None:
                unsch_tasks = np.array(env.get_unscheduled_tasks(), dtype=np.int64)
                valid_tasks = np.array(env.get_valid_tasks(t), dtype=np.int64)

                if len(valid_tasks) > 0:
                    g = build_hetgraph(env.halfDG, env.num_tasks, env.num_robots, env.dur,
                                       map_width, np.array(env.loc, dtype=np.int64),
                                       1.0, env.partials, unsch_tasks, robot_chosen,
                                       valid_tasks)
                    g = g.to(device)

                    feat_dict = hetgraph_node_helper(env.halfDG.number_of_nodes(),
                                                     env.partialw,
                                                     env.partials, env.loc, env.dur,
                                                     map_width, env.num_robots,
                                                     len(valid_tasks))

                    feat_dict_tensor = {}
                    for key in feat_dict:
                        feat_dict_tensor[key] = torch.Tensor(feat_dict[key]).to(device)

                    task_chosen = gnn_pick_task(g, valid_tasks, policy_net,
                                                feat_dict_tensor)

                    if task_chosen >= 0:
                        task_dur = env.dur[task_chosen-1][robot_chosen]
                        rt, reward, done = env.insert_robot(task_chosen, robot_chosen)
                        robots.update_status(task_chosen, robot_chosen, task_dur, t)
                        #print(task_chosen, robot_chosen, task_dur)
                        '''
                        Check for termination
                        '''
                        if not rt:
                            print('Infeasible after %d insertions' % (len(env.partialw)-1))
                            results.append([graph_no, -1])
                            terminate = True
                            break
                        elif env.partialw.shape[0] == (env.num_tasks+1):
                            feas_count += 1
                            dqn_opt = env.min_makespan
                            print('Feasible solution found, min makespan: %f'
                                  % env.min_makespan)
                            results.append([graph_no, dqn_opt])
                            terminate = True
                            break

                        # Attempt to pick another robot
                        robot_chosen = robots.pick_robot_by_min_dur(t, env, args.version, exclude)
                    else:
                        # No valid tasks for this robot, move to next
                        exclude.append(robot_chosen)
                        robot_chosen = robots.pick_robot_by_min_dur(t, env, args.version, exclude)
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
