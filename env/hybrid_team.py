"""
Created on Mon Sep  6 13:08:23 2021

@author: baltundas3

Hybrid Team

"""

import sys
import math
# setting path
sys.path.append('../')
from env.worker import *
from env.human_learning_rate import HumanLearningRate
from env.mrc_problem import MRCProblem

'''
Task class
'''
class Task(object):
    def __init__(self, t_id, s_time, e_time):
        self.id = t_id
        self.start_time = s_time
        self.end_time = e_time

class TaskEstimator(object):
    # TODO: Replace this with a Kalman Filter
    def __init__(self):
        self.std_c = 15.0
        self.std_k = 15.0
        self.std_beta = 5.0 / 3
        pass
        
    def estimate(self, actual_time, repeat_count, noise):
        # c_mean = initial_time / 2
        # k_mean = initial_time - c_mean
        # beta_mean = 5.0
        # if noise:
        #     c = np.random.normal(c_mean, self.std_c)
        #     k = np.random.normal(k_mean, self.std_k)
        #     beta_sample = np.random.normal(beta_mean, self.std_beta)
        #     return min(max(10, c + k * math.exp(-1 * beta_sample * repeat)), 100) # clip to known duration range
        # else:
        #     return min(max(10, c_mean + k_mean * math.exp(-1 * beta_mean * repeat)), 100) # clip to known duration range
        error_std = 15* math.exp(-0.5 * repeat_count)
        return max(0, np.random.normal(actual_time, error_std))

"""
Hybrid team class
"""
class HybridTeam(object):
    def __init__(self, problem: MRCProblem):

        self.num_tasks = problem.num_tasks
        self.num_robots = problem.num_robots
        self.num_humans = problem.num_humans
        self.human_learning_rate = problem.human_learning_rate
        self.task_estimator = TaskEstimator()
        
        self.dur = problem.dur
        self.humans_sampled = False
        # [0 : num_robots] -> Robots
        # [num_robots : num_robots + num_humans] -> Humans
        self.workers = [Robot(i) for i in range(self.num_robots)] \
                        + [Human(i, self.human_learning_rate) for i in range(self.num_robots, self.num_robots + self.num_humans)]
        self.calibrate_tasks(problem)

    def calibrate_tasks(self, problem: MRCProblem):
        # robots have constant time
        for i in range(self.num_robots):
            self.workers[i].set_tasks(problem.dur[:, i])
    
    def sample_humans(self):
        for task in range(self.num_tasks):
            for wid in range(self.num_robots, self.num_robots + self.num_humans):
                self.dur[task][wid] = self.get_duration(task, wid)
        self.humans_sampled = True

    def get_worker(self, w_id : int, w_type : WorkerType = None):
        assert 0 <= w_id < self.__len__()
        if w_type == None:
            return self.workers[w_id]
        elif w_type == WorkerType.ROBOT:
            assert(self.num_robots > r_id)
            return workers[r_id]
        elif w_type == WorkerType.HUMAN:
            assert (self.num_robots > h_id)
            return workers[self.num_robots + h_id]
    
    def get_duration(self, task_id: int, w_id: int, estimator = False, noise = False):
        """Generates the Durations it would take for a task_id to complete w_id

        Args:
            task_id (int): ID of the Task being completed to
            w_id (int): ID of the Task completor
            estimator (bool, optional): Determines whether or not task duration is selected from the estimator. Defaults to False.
            noise (bool, optional): the noise presence in the task duration. Defaults to False.

        Returns:
            int: duration of task completion
        """
        assert 0 <= w_id < self.__len__()
        d = 100 # Worst case scenerio
        if not self.humans_sampled:
            d = self.workers[w_id].get_duration_of_task(task_id)
        else: # return from presampled human samples
            #task_count = self.workers[w_id].task_counter[task_id]
            # return self.human_sample[task_id, w_id - self.num_robots, task_count]
            d = self.dur[task_id][w_id]
        if estimator and self.workers[w_id].type == WorkerType.HUMAN:
            if task_id not in self.workers[w_id].task_counter:
                self.workers[w_id].task_counter[task_id] = 0
            return self.task_estimator.estimate(d, self.workers[w_id].task_counter[task_id], noise)
        return d
        
    def available_workers(self, timepoint):
        """return a list of workers that are available at a given timepoint

        Args:
            timepoint (int): timepoint

        Returns:
            list(ids): list of ids of all the available workers in the given timepoint, returns [] if no robot is available.
        """
        available = []
        for i in range(self.num_robots + self.num_humans):
            if self.workers[i].next_available_time <= timepoint:
                available.append(self.workers[i].id)
        return available

    def __len__(self):
        return len(self.workers)
    
    """Update the Status of a Worker
    
    Args:
        task_chosen (int): TaskID
        worker_chosen (int): WorkerID
        task_dur (int): duration of the task
        t (int): start time
    """
    # Update the status of worker after scheduling the chosen task
    def update_status(self, task_chosen, worker_chosen, task_dur, t):
        self.workers[worker_chosen].schedule.append(Task(task_chosen, t, t + task_dur))
        self.workers[worker_chosen].next_available_time = t + task_dur  

    # print all workers' schedules
    def print_schedule(self):
        for i in range(self.num_robots + self.num_humans):
            print('Worker %d, Type %s' % self.workers[i].id, self.workers[i].type)
            for task in self.workers[i].schedule:
                print('Task (%d,%d,%d)'%(task.id, task.start_time, task. end_time))
                
    def reset(self):
        for worker in self.workers:
            worker.reset()

    # from env.scheduling_env import SchedulingEnv
    def pick_worker_by_min_dur(self, time, env, version, exclude=[]):
        """Returns the worker with minimum average duration on unscheduled tasks for v1,
        min duration on any one unscheduled task for v2,
        min average duration on valid tasks for v3
        """
        dur_and_workers = []  # List of (average duration, robot id) tuples

        if version == 'v3':
            tasks = env.get_valid_tasks(time)
        else:
            tasks = env.get_unscheduled_tasks()
        if len(tasks) == 0:
            return None
        for i in range(self.num_robots + self.num_humans):
            if self.workers[i].id not in exclude:
                if self.workers[i].next_available_time <= time:
                    dur = env.get_duration_on_tasks(self.workers[i].id, tasks)
                    if version == 'v2':
                        dur_and_workers.append((min(dur), self.workers[i].id))
                    else:
                        dur_and_workers.append((sum(dur) / len(dur), self.workers[i].id))

        # No robot is available
        if len(dur_and_workers) == 0:
            return None

        return min(dur_and_workers)[1]

# '''
# Pick a task that has the earlist deadline
#     minDG: APSP graph
#     act_task: unscheduled tasks
# '''
# def pick_task(minDG, act_task, timepoint):
#     length = len(act_task)
#     if length == 0:
#         return -1
    
#     tmp = np.zeros(length, dtype=np.float32)
    
#     for i in range(length):
#         ti = act_task[i]
#         #si = 's%03d' % ti
#         fi = 'f%03d' % ti
#         # pick the task with the earlist possible finish time
#         tmp[i] = -1.0 * minDG[fi]['s000']['weight']
    
#     idx = np.argmin(tmp)
#     task_chosen = act_task[idx]
    
#     sk = 's%03d' % task_chosen
#     time_sk = -1.0 * minDG[sk]['s000']['weight']
#     if time_sk <= timepoint:
#         return task_chosen
#     else:
#         return -1
