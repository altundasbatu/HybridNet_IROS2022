# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 14:58:41 2020

@author: pheno and baltundas3

Integrated Human Model

Utility functions

1. Replace floyd_warshall with Johnson's for STN preprocessing
"""


import copy
import random
from collections import Counter
from collections import namedtuple

import dgl
import networkx as nx
import numpy as np
import torch

from benchmark.JohnsonUltra import johnsonU

from env.mrc_problem import MRCProblem
from env.scheduling_env import SchedulingEnv
from env.hybrid_team import HybridTeam

from env.multi_round_scheduling_env import MultiRoundSchedulingEnv

def build_hetgraph(halfDG, num_tasks, num_robots, num_humans, dur, 
                    partials, unsch_tasks):
    """
    Helper function for building HetGraph
    Q nodes are built w.r.t selected_worker & unsch_tasks
        valid_tasks: available tasks filtered from unsch_tasks
        
    Args:
        loc_dist_threshold: Distance threshold for two locations to be connected by an edge
    """
    num_workers = num_robots + num_humans
    # num_locs = map_width * map_width
    num_values = len(unsch_tasks)
    num_nodes_dict = {'task': num_tasks + 2,
                      'worker': num_workers,
                      'state': 1}
    # print(num_nodes_dict)
    # Sort the nodes and assign an index to each one
    task_name_to_idx = {node: idx for idx, node in enumerate(sorted(halfDG.nodes))}
    task_edge_to_idx = {(from_node, to_node): idx for idx, (from_node, to_node) in enumerate(halfDG.edges)}

    # Workers:

    # List of (task id, worker id) tuples
    task_to_worker_data = []

    for wj in range(num_workers):
        # add f0
        task_to_worker_data.append((0, wj))
        # add si (including s0)
        for i in range(len(partials[wj])):
            ti = partials[wj][i].item()
            task_id = ti + 1
            task_to_worker_data.append((task_id, wj))

    unsch_task_to_worker = []
    for wj in range(num_workers):
        for t in unsch_tasks:
            task_id = t + 1
            unsch_task_to_worker.append((task_id, wj))
    # print(unsch_task_to_worker)
    worker_com_data = [(i, j) for i in range(num_workers) for j in range(num_workers)]

    #
    data_dict = {
        ('task', 'temporal', 'task'): (
            # Convert named edges to indexes
            [task_name_to_idx[from_node] for from_node, _ in halfDG.edges],
            [task_name_to_idx[to_node] for _, to_node in halfDG.edges],
        ),
        ('task', 'assigned_to', 'worker'): (
            [task for task, _ in task_to_worker_data],
            [worker for _, worker in task_to_worker_data],
        ),
        ('task', 'take_time', 'worker'): (
            [task for task, _ in unsch_task_to_worker],
            [worker for _, worker in unsch_task_to_worker],
        ),
        ('worker', 'use_time', 'task'): (
            [worker for _, worker in unsch_task_to_worker],
            [task for task, _ in unsch_task_to_worker],
        ),
        ('worker', 'com', 'worker'): (
            [i for i, _ in worker_com_data],
            [j for _, j in worker_com_data],
        ),
        # 4. Add graph summary nodes
        # [task] — [in] — [state]
        ('task', 'tin', 'state'): (
            list(range(num_tasks + 2)),
            np.zeros(num_tasks + 2, dtype=np.int64),
        ),
        # [worker] — [in] — [state]
        ('worker', 'win', 'state'): (
            list(range(num_workers)),
            np.zeros(num_workers, dtype=np.int64),
        ),
        # [state] — [in] — [state] self-loop
        ('state', 'sin', 'state'): (
            [0],
            [0],
        )
    }
    # print(data_dict, num_nodes_dict)
    graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict, idtype=torch.int64)
    # Store data of edges by index, as DiGraph.edges.data does not guarantee to have exactly the same
    # ordewing as Digraph.edges
    temporal_edge_weights = torch.zeros((len(halfDG.edges), 1), dtype=torch.float32)
    # Unpack indexes of edge weights
    weights_idx = [task_edge_to_idx[from_node, to_node] for from_node, to_node, _ in halfDG.edges.data('weight')]
    # Put weights in tensor according to their indexes
    temporal_edge_weights[weights_idx, :] = torch.tensor([[weight] for _, _, weight in halfDG.edges.data('weight')],
                                                      dtype=torch.float32)
    graph.edges['temporal'].data['weight'] = temporal_edge_weights

    takes_time_weight = torch.zeros((len(unsch_task_to_worker), 1), dtype=torch.float32)
    for idx, (task, worker) in enumerate(unsch_task_to_worker):
        # Subtract 2 because task 1's node id is 2, but has index 0 in dur
        takes_time_weight[idx] = dur[task - 2, worker]
    graph.edges['take_time'].data['t'] = takes_time_weight
    # Ordewing of takes_time and uses_time edges are exactly the same
    graph.edges['use_time'].data['t'] = takes_time_weight.detach().clone()

    return graph

def hetgraph_node_helper(number_of_nodes, curr_partialw, curr_partials,
                         durations, num_workers, num_values):
    """
    Generate initial node features for hetgraph
    The input of hetgraph is a dictionary of node features for each type
    Args:
        number_of_nodes: number of nodes in half distance graph (halfDG)
        curr_partialw: partial solution/whole
        curr_partials: partial solution/seperate
        locations: np array locations
        durations: np array task durations
        num_workers: number of workers
        num_values: number of actions / Q values
    Returns:
        feat_dict: node features stored in a dict
    """
    feat_dict = {}
    num_locations = 0 # map_width * map_width

    # Task features.
    # For scheduled tasks, the feature is [1 0 dur 0 dur 0]
    # For unscheduled ones, the feature is [0 1 min max-min mean std]
    # print(number_of_nodes)
    feat_dict['task'] = np.zeros((number_of_nodes, 6))
    # print(durations)
    max_dur, min_dur = durations.max(axis=1), durations.min(axis=1)
    mean_dur, std_dur = durations.mean(axis=1), durations.std(axis=1)

    # f0
    feat_dict['task'][0, 0] = 1

    # s0~si. s0 has index 1
    for i in range(1, number_of_nodes):
        ti = i-1
        if ti in curr_partialw:
            feat_dict['task'][i, 0] = 1
            if ti > 0:
                # Ignore s0
                for j in range(num_workers):
                    if ti in curr_partials[j]:
                        rj = j
                        break              
                
                feat_dict['task'][i, [2, 4]] = durations[ti-1][rj]
        else:
            feat_dict['task'][i] = [0, 1, min_dur[ti-1], max_dur[ti-1] - min_dur[ti-1], 
                                    mean_dur[ti-1], std_dur[ti-1]]
    
    # # [loc]
    # feat_dict['loc'] = np.zeros((num_locations, 1))
    # serialized_locs = [(locations[i, 1] - 1) * map_width + locations[i, 0] - 1 for i in range(locations.shape[0])]
    # loc_counter = Counter(serialized_locs)
    # for i in range(num_locations):
    #     # number of tasks in location
    #     feat_dict['loc'][i, 0] = loc_counter[i]
    
    # [worker]
    feat_dict['worker'] = np.zeros((num_workers, 1))
    for i in range(num_workers):
        # number of tasks assigned so far
        # including s0
        feat_dict['worker'][i, 0] = len(curr_partials[i])
    
    # [state]
    feat_dict['state'] = np.array((number_of_nodes-1, len(curr_partialw),
                                   num_locations, num_workers)).reshape(1,4)
    
    return feat_dict


'''
Transition for n-step
state
    curr_g: networkx graph updated with current solution
    curr_partials: partial solution as a list of numpy arrays (int)
        [sd0 sd1 ...]
            sd0: partial schedule of robot 0
            sd1: partial schedule of robot 1
            ......
    curr_partialw: partial schedule of all tasks selected
    locations: the location of each task
    durations: the duration of each task
action
    act_task: ti
    act_robot: rj
        append ti to rj's partial schedule
reward
    reward_n: total future discounted rewards
state after 1-step
    next_g: networkx graph
    next_partial: next partial solution
termination
    next_done: if True, means the next state is a termination state
        one episode finishes
        1. finish with feasible solution
        2. stop with infeasible partial
'''
Transition = namedtuple('Transition',
                        ('env_id', 'curr_g', 'curr_partials', 'curr_partialw',
                        #  'locs', 
                         'durs',
                         'act_task', 'act_robot',
                         'reward_n', 'next_g', 'next_partials',
                         'next_partialw', 'next_done'))

'''
Replay buffer
'''
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # Saves a transition
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # print("Memory", self.memory, batch_size)
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

'''
Enumerate all possible insertions (rollout version) based on
    num_tasks: number of total tasks 1~N
    curr_partialw: partial solution
Return
    act_task: list of all possible insertions
'''
def action_helper_rollout(num_tasks, curr_partialw):
    act_task = []
    # pick a task t_i from {unallocated}
    for i in range(1, num_tasks + 1):
        if i not in curr_partialw:
            act_task.append(i)

    return np.array(act_task)

if __name__ == '__main__':
    # problem path
    fname = 'env/tmp/test_file'

    problem = MRCProblem(fname=fname)
    team = HybridTeam(problem)
    # initialize env
    env = SchedulingEnv(problem, team)
    # env.g is the original STN
    print(env.graph.nodes())
    print(env.graph.number_of_edges())
    # env.halfDG is the simplified graph to be used for graph construction
    print(sorted(env.halfDG.nodes()))
    print(env.halfDG.number_of_edges())
    
    # load solution
    # problem.get_solution()
    if problem.optimal_schedule is None:
        problem.get_optimal_with_gurobi(fname+'_gurobi.log', threads=2)
        
    optimals = problem.optimal_schedule[0]
    optimalw = problem.optimal_schedule[1]
    
    print(env.problem.optimal_schedule)
    
    print('Initial makespan: ', env.min_makespan)
    # check gurobi solution
    rs = []
    for i in range(len(optimalw)):
        for j in range(len(env.team)):
            if optimalw[i] in optimals[j]:
                rj = j
                break
        task_id = optimalw[i]
        rt, reward, done, arg = env.step(task_id, rj)
        rs.append(reward)
        print('Insert %d, %d' % (optimalw[i], rj))
        print('No. Edges: %d' % env.halfDG.number_of_edges())
        print('Returns: ', rt, reward, done, env.min_makespan)
        if not rt:
            print('Infeasible!')
            break
        
    print(env.partialw)
    print(sum(rs))
    print('test passed')

    g = build_hetgraph(env.halfDG, problem.num_tasks, problem.num_robots, problem.num_humans, env.dur, env.partials, env.get_unscheduled_tasks())
    print(g)

