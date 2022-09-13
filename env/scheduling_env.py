"""
Created on Mon Sep  6 11:43:13 2021

@author: baltundas3

ScheduleNet Environment based on OpenAI Code
"""


import bisect
import datetime
import random
import copy

import pandas as pd
import gym
import numpy as np
from pathlib import Path


import networkx as nx

import sys
# setting path
sys.path.append('../')
from benchmark.JohnsonUltra import johnsonU
from env.mrc_problem import MRCProblem
from env.worker import *
from env.hybrid_team import HybridTeam


"""
OpenAI Based Scheduling Environment
"""
class SchedulingEnv(gym.Env):
    
    def __init__(self, fname: str = None, 
                        problem: MRCProblem = None, 
                        team: HybridTeam = None, 
                        restrict_same_time_scheduling: bool = False,
                        infeasible_coefficient: float = 1.0,
                        noise: bool = False,
                        sample_humans = True):
        """Single Round Scheduling Environment using OpenAI Gym Environment used for simulation and testing of the ScheduleNet Architecture and Variants.

        Args:
            problem (MRCProblem): Scheduling Problem
            team (HybridTeam): Scheduling Team
        """
        if fname == None:
            self.problem = problem
            self.problem.noise = noise
            self.team = team
            self.team.noise = noise
        else:
            self.problem = MRCProblem(fname=fname, max_deadline_multiplier=infeasible_coefficient, noise = noise)
            self.team = HybridTeam(self.problem)
        if sample_humans:
            self.team.sample_humans()
        # Deep Copy the graph, dur and locs
        self.graph = copy.deepcopy(self.problem.DG)
        self.dur = copy.deepcopy(self.team.dur.copy())
        # self.locs = copy.deepcopy(self.problem.locs.copy())
        self.restrict_same_time_scheduling = restrict_same_time_scheduling
        
        self.C = 3.0 # discount factor for reward calculation

        self.partials = []
        for i in range(len(self.team)):
            self.partials.append(np.zeros(1, dtype=np.int32))

        self.partialw = np.zeros(1, dtype=np.int32)

        self.g = self.problem.initialize_STN()
        # get initial min make span
        self.halfDG = None
        self.success, min_makespan = self._check_consistency_makespan()
        if self.success:
            self.min_makespan = min_makespan
        else:
            print('Initial STN infeasible.')

    def _check_consistency_makespan(self, updateDG = True):
        """Check consistency and get min make span
        Also creates the half min graph
        Args:
            updateDG (bool): [description]. Defaults to True.

        Returns:
            consistent (bool): Consistency of the graph
            min_makespan (float): Minimum Makespan of the graph
        """
        consistent = True
        try:
            p_ultra, d_ultra = johnsonU(self.graph)
        except Exception as e:
            consistent = False
            # print('Infeasible:', e)
                
        '''
        Makespan
        Only consider the last finish time of scheduled tasks
        '''
        if consistent:        
            if len(self.partialw) == 1:
                min_makespan = 0.0
            else:
                tmp = []
                for i in range(1,len(self.partialw)):
                    ti = self.partialw[i]
                    fi = 'f%03d' % ti
                    tmp.append(-1.0 * d_ultra[fi]['s000'])
    
                tmp_np = np.array(tmp)
                min_makespan = tmp_np.max()
        else:
            min_makespan = self.problem.max_deadline
            return consistent, min_makespan
        
        if not updateDG:
            return consistent, min_makespan
        
        '''
        Min distance graph & Half min graph
        '''
        juDG = nx.DiGraph()
        for i in range(0, self.problem.num_tasks+1):
            # Add si and fi
            si = 's%03d' % i
            fi = 'f%03d' % i
            # minDG.add_nodes_from([si, fi])
            if i == 0:
                juDG.add_nodes_from([si, fi])
            else:
                juDG.add_node(si)
        
        # add shortest path distance edges
        for k_start in d_ultra:
            for k_end in d_ultra[k_start]:
                #print(key_start, key_end)
                # check if path is inf
                if d_ultra[k_start][k_end] < 9999:
                    # minDG.add_edge(k_start, k_end, 
                    #                weight = d_ultra[k_start][k_end])
                    if juDG.has_node(k_start) and juDG.has_node(k_end):
                        juDG.add_edge(k_start, k_end,
                                      weight = d_ultra[k_start][k_end])
        
        # self.minDG = minDG
        self.halfDG = juDG
        
        return consistent, min_makespan
    
    def reset(self):
        # clear and get a new copy of the task environment
        self.graph = copy.deepcopy(self.problem.DG)
        self.dur = copy.deepcopy(self.team.dur.copy())
        # self.locs = copy.deepcopy(self.problem.locs.copy())
        
        self.partials = []
        for i in range(len(self.team)):
            self.partials.append(np.zeros(1, dtype=np.int32))

        self.partialw = np.zeros(1, dtype=np.int32)

    def step(self, ti: int, wi: int, diff = 1.0, updateDG = True):
        """Step Action for the Scheduler, appends the task_id to the worker_id's partial schedule and updates the self.graph

        Args:
            ti (int): Task ID [1, num_tasks] 
            wi (int): Worker ID [0. num_robots-1]
            diff (float): Allowed time distance. Defaults to 1.0.
            updateDG (bool, optional): [description]. Defaults to True.

        Returns:
            state (): Current Schedule
            reward (float)
            done (boolean)
            info
        """
        # sanity check
        if ti <= 0 or ti > self.problem.num_tasks:
            print("No Task ID:", ti)
            return False
        if wi < 0 or wi >= len(self.team):
            print("No Worker:", wi)

        # find tj and update partial solution
        # tj is the last task of wi's partial schedule
        # insert ti right after tj
        tj = self.partials[wi][-1]
        self.partials[wi] = np.append(self.partials[wi], ti)
        self.partialw = np.append(self.partialw, ti)

        # update graph
        # insert ti after tj, no need to add when tj==0    
        # no need to insert if a wait constraint already exists
        if tj != 0:
            si = 's%03d' % ti
            fj = 'f%03d' % tj
            if not self.graph.has_edge(si, fj):
                self.graph.add_edge(si, fj, weight = 0)
        
        '''
        [New] Also, replace the task duration of ti with actual duration
        '''
        si = 's%03d' % ti
        fi = 'f%03d' % ti
        ti_dur = self.team.get_duration(ti-1, wi) # Let the team provide how long it takes.
        # this will rewrite previous edge weights
        self.graph.add_weighted_edges_from([(si, fi, ti_dur),
                                        (fi, si, -1 * ti_dur)])
        
        if self.restrict_same_time_scheduling:
            '''
            make sure the start time of all unscheduled tasks is no earlier than si
            '''
            for k in range(1, self.problem.num_tasks+1):
                if k not in self.partialw:
                    # tk starts no earlier than si
                    # si <= sk, si-sk<=0, sk->si:0
                    si = 's%03d' % ti
                    sk = 's%03d' % k
                    if not self.graph.has_edge(sk, si):
                        self.graph.add_edge(sk, si, weight = 0)

        # '''
        # make sure the start time of all unscheduled tasks that
        # are within the allowed distance (diff) happen after fi
        # '''
        # for k in range(1, self.problem.num_tasks+1):
        #     if k not in self.partialw:
        #         xi, yi = self.locs[ti-1]
        #         xk, yk = self.locs[k-1]
        #         dist_2 = (xi - xk) * (xi - xk) + (yi - yk) * (yi - yk)               
                
        #         if dist_2 <= diff * diff:
        #             # tk starts after fi
        #             # fi <= sk, fi-sk <=0, sk->fi:0
        #             fi = 'f%03d' % ti
        #             sk = 's%03d' % k
        #             if not self.g.has_edge(sk, fi):
        #                 self.g.add_edge(sk, fi, weight=0)

        # calculate reward for this insertion
        success, reward = self._calc_reward_discount(updateDG)
        # check done/termination
        if success==False:
            done = True
        elif (self.partialw.shape[0]==self.problem.num_tasks+1):
            done = True
        else:
            done = False
        
        return success, reward, done, ""

    def _calc_reward_discount(self, updateDG = True):
        """Reward R of a state-action pair is defined as the change
                in objective values after taking the action,
            R = −1 × (Zt+1 − Zt).
            divide Zt by a factor D > 1 if xt is not a termination state

            Z(infeasible) = problem.max_deadline

        Args:
            updateDG (bool, optional): [description]. Defaults to True.

        Returns:
            success (bool): [description]
            reward (float): [description]
        """
    
        success, min_makespan = self._check_consistency_makespan(updateDG)
        # feasible
        if success:
            # if last step
            if self.partialw.shape[0]==(self.problem.num_tasks+1):
                delta = min_makespan - self.min_makespan/self.C
            # discounted delta
            else:
                delta = (min_makespan - self.min_makespan)/self.C
        # infeasible
        else:
            delta = self.problem.max_deadline - self.min_makespan/self.C
            min_makespan = self.problem.max_deadline
            # print("Infeasible Step Taken", delta, min_makespan)
        
        reward = -1.0 * delta
        
        self.min_makespan = min_makespan
        return success, reward

    def get_infeasible_reward(self, remaining_tasks):
        infeasible_makespan = self.problem.get_max_makespan(remaining_tasks)
        delta = infeasible_makespan / self.C
        reward = -1.0 * delta
        # TODO: Include human learning here
        return reward

    def pick_worker_by_min_dur(self, time, version, exclude=[]):
        """Returns the worker with minimum average duration on unscheduled tasks for v1,
        min duration on any one unscheduled task for v2,
        min average duration on valid tasks for v3
        """
        dur_and_worker = []  # List of (average duration, robot id) tuples

        if version == 'v3':
            tasks = self.get_valid_tasks(time)
        else:
            tasks = self.get_unscheduled_tasks()
        if len(tasks) == 0:
            return None

        for worker in self.team.workers:
            if worker.id not in exclude:
                if worker.next_available_time <= time:
                    dur = self.get_duration_on_tasks(worker.id, tasks)
                    # print(tasks, dur, worker.id)
                    if version == 'v2':
                        dur_and_worker.append([min(dur), worker.id])
                    else:
                        dur_and_worker.append([sum(dur) / len(dur), worker.id])
        
        # No worker is available
        if len(dur_and_worker) == 0:
            return None
        # print(dur_and_worker)
        return min(dur_and_worker)[1]
        # return int(np.min(np.array(dur_and_worker), axis=0)[1])
    
    def get_unscheduled_tasks(self):
        """Returns unscheduled tasks given partialw

        Returns:
            unsch_tasks (list(int)): a list of unscheduled tasks
        """
        unsch_tasks = []
        for i in range(1, self.problem.num_tasks+1):
            if i not in self.partialw:
                unsch_tasks.append(i)
        
        return np.array(unsch_tasks)

    def get_duration_on_tasks(self, worker_id, tasks):
        """Returns durations of a robot on a list of tasks.
        Task ids should be 1-indexed, and robot id should be 0-indexed
        Args:
            worker_id (int): WorkerID
            tasks (list(int)): List of Tasks
        Returns:
            (list): List of Times for the tasks
        """
        assert min(tasks) > 0, 'Tasks should be 1-indexed'
        assert 0 <= worker_id < len(self.team), 'Robot should be 0-indexed'
        
        worker = self.team.get_worker(worker_id)
        return [worker.get_duration_of_task(task - 1) for task in tasks]
        
    def get_valid_tasks(self, timepoint):
        '''
        Return unscheduled tasks given partialw
            plus checking if the task can starts at current timepoint
        '''
        valid_tasks = []
        for i in range(1, self.problem.num_tasks+1):
            if i not in self.partialw:
                # check task start time
                # si->s0: A
                # s0 - si <= A
                # si >= -A
                si = 's%03d' % i
                time_si = -1.0 * self.halfDG[si]['s000']['weight']
                # time_si is the earliest time task i can happen
                if time_si <= timepoint:
                    valid_tasks.append(i)
        
        return np.array(valid_tasks)
    
    def get_rSTN(self, worker_chosen, valid_task):
        """Return an updated min worker STN
        with task duration (valid unscheduled tasks) 
        replaced with the task duration of chosen robot
        plus consistency check

        Args:
            worker_chosen (int): WorkerID of the chosen worker
            valid_task (list): List of Valid Tasks
        Returns:
            [type]: [description]
        """
        rSTN = copy.deepcopy(self.g)
        # modify STN
        for i in range(len(valid_task)):
            ti = valid_task[i]
            si = 's%03d' % ti
            fi = 'f%03d' % ti
            ti_dur = self.dur[ti-1][worker_chosen]
            rSTN.add_weighted_edges_from([(si, fi, ti_dur),
                                          (fi, si, -1 * ti_dur)])       
        
        # check consistency
        consistent = True    
        try:
            p_ultra, d_ultra = johnsonU(rSTN)
        except Exception as e:
            consistent = False
            # print('Infeasible:', e) 

        if consistent:    
            # get min STN
            min_rSTN = nx.DiGraph()
            for i in range(0, self.problem.num_tasks+1):
                # Add si and fi
                si = 's%03d' % i
                fi = 'f%03d' % i
                min_rSTN.add_nodes_from([si, fi])
            
            # add shortest path distance edges
            for k_start in d_ultra:
                for k_end in d_ultra[k_start]:
                    # check if path is valid
                    if d_ultra[k_start][k_end] < 9999:
                        min_rSTN.add_edge(k_start, k_end, 
                                       weight = d_ultra[k_start][k_end])        
            
            return min_rSTN, True
        else:
            return None, False

    def render(self):
        """[OPTIONAL]
        """
        pass

if __name__ == '__main__':
    fname = "tmp/test_file"
    # problem = MRCProblem(fname=fname)
    # team = HybridTeam(problem)
    # initialize env
    env = SchedulingEnv(fname)
    # env.graph is the original STN
    print(env.graph.nodes())
    print(env.graph.number_of_edges())
    # env.halfDG is the simplified graph to be used for graph construction
    print(sorted(env.halfDG.nodes()))
    print(env.halfDG.number_of_edges())

    # load solution
    # problem.get_solution()
    if env.problem.optimal_schedule is None:
        env.problem.get_optimal_with_gurobi(fname+'_gurobi.log', threads=2)
        
    optimals = env.problem.optimal_schedule[0]
    optimalw = env.problem.optimal_schedule[1]
    
    # optimal, passed = problem.get_optimal_with_gurobi("tmp/test_file_gurobi.log", threads=1)
    #optimalw[8] = 12
    #optimalw[9] = 7
    
    print('Initial makespan: ', env.min_makespan)
    # check gurobi solution
    # print(optimalw, optimals)
    rs = []
    for i in range(len(optimalw)):
        for j in range(len(env.team)):
            if optimalw[i] in optimals[j]:
                wj = j
                break
        task_id = optimalw[i]
        rt, reward, done, arg = env.step(task_id, wj)
        rs.append(reward)
        print('Insert %d, %d' % (optimalw[i], wj))
        print('No. Edges: %d' % env.halfDG.number_of_edges())
        print('Returns: ', rt, reward, done, env.min_makespan)
        if not rt:
            print('Infeasible!')
            break
        
    print(env.partialw)
    print(sum(rs))
    print('test passed')
    
    problem = env.problem
    team = env.team
    env2 = SchedulingEnv(problem=problem, team=team)