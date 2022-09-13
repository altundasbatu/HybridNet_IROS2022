"""
Created on Mon Sep  6 12:08:23 2021

@author: baltundas3

MRC Problem
"""
import random
import networkx as nx
import numpy as np
import errno
import os

from glob import glob

import numpy as np

import sys
# setting path
sys.path.append('../')
from gen.gmodel import GModel
# sys.path.append('../')
from env.human_learning_rate import HumanLearningRate

class MRCProblem(object):
    """MRC Problem
    ```python
    # Create new instance
    prob1 = MRCProblem(num_tasks=10, num_robots=5, num_humans=5, map_width=5)
    # Read from a file
    prob2 = MRCProblem(fname="somefname")
    ```
    """
    def __init__(self, 
                    num_tasks = 20, 
                    num_robots = 5, num_humans = 1, 
                    map_width = 5, fname = None,
                    min_dur = 15, max_dur = 100,
                    max_deadline_multiplier = 2.0,
                    prob_deadline = 0.25, prob_wait_ori = 0.25,
                    noise: bool = False):
        """MRC Problem Initializer

        Args:
            num_tasks (int): Number of tasks. Defaults to 20.
            num_robots (int): Number of robots. Defaults to 5.
            num_humans (int): Number of Humans. Defaults to 1.
            map_width (int): Width of the Map. Defaults to 5.
            fname (str): [description]. Defaults to None.
        """
        self.max_deadline_multiplier = max_deadline_multiplier

        self.sol = None
        self.optimal = None
        self.optimal_schedule = None
        self.noise = noise
        if fname == None:
            self._generate_problem(num_tasks, num_robots, num_humans, map_width, min_dur, max_dur, prob_deadline, prob_wait_ori)
        else:
            self._read_from_file(fname)
    
    def refactor_problem_from_team(self, team, estimate = False, noise = False):
        """Refactors the Environment Problem based on the current team model with task completion counts

        Args:
            team (HybridTeam): Team of Human and Robot Agents
            estimate (bool, optional): Boolean to determine if the sampling is done by the estimator or the actual human model. Defaults to False.
            noise (bool, optional): presence of noise within the sampling. Defaults to False.
        """
        self.dur = np.zeros((self.num_tasks, self.num_robots + self.num_humans), dtype=np.int32)
        for i in range(self.num_tasks):
            for j in range(self.num_robots + self.num_humans):
                self.dur[i][j] = team.get_duration(i, j, estimate, noise)
        # self.set_max_makespan()
        # self.DG = nx.DiGraph()
        # self._initialize_graph()
        # self.sol = None
        # self._generate_constraints()
        # ready = self.check_consistency()
        # print("Ready:", ready)
        
    # Generate Problem from Parameters
    def _generate_problem(self,
                            num_tasks: int, 
                            num_robots: int, 
                            num_humans: int,
                            map_width: int,
                            min_dur: int, 
                            max_dur: int,
                            prob_deadline: float, 
                            prob_wait_ori: float):
                            # decrease probabilities

        self.num_tasks = num_tasks
        
        self.num_robots = num_robots
        self.num_humans = num_humans

        self.min_dur = min_dur
        self.max_dur = max_dur
        
        self.map_width = map_width
    
        self.human_learning_rate = HumanLearningRate(self.num_tasks, self.num_robots, self.num_humans, self.min_dur, self.max_dur)
        self._generate_durations()
        self.set_max_makespan()
        
        self.DG = nx.DiGraph()
        
        self._initialize_graph()

        self.prob_deadline = prob_deadline
        self.prob_wait = prob_wait_ori / (self.num_tasks-1)

        self.sol = None

        self._generate_constraints()
        ready = self.check_consistency()
        # print(len(list(self.DG.edges)))
    
        while not ready:
            self._reinitialize()
            self._generate_constraints()
            ready = self.check_consistency()
            # print(len(list(self.DG.edges)))
            # print(ready)
        # print(ready)

    def _initialize_graph(self):
        # Constraints
        # self.dur = np.zeros((self.num_tasks, self.num_robots + self.num_humans), dtype=np.int32)
        self.ddl = []
        self.wait = []
        
        # Initialize directed graph        
        self.DG.add_nodes_from(['s000', 'f000'])
        self.DG.add_edge('s000', 'f000', weight = self.max_deadline)
        
        # Add tasks
        for i in range(1, self.num_tasks+1):
            # Add si and fi
            si = 's%03d' % i
            fi = 'f%03d' % i
            self.DG.add_nodes_from([si, fi])
            self.DG.add_weighted_edges_from([(si, 's000', 0),
                                             ('f000', fi, 0)])

    def _generate_constraints(self):
        self._generate_durations_graph()
        self._generate_ddl()
        self._generate_wait()
        self._generate_locs()
        
    # Task durations
    def _generate_durations(self):
        self.dur = np.zeros((self.num_tasks, self.num_robots + self.num_humans), dtype=np.int32)
        for i in range(self.num_tasks):
            mean = random.randint(1, 10)
            gap = random.randint(1, 3)
            human_vals = self.human_learning_rate.task_models[i,:,0] + self.human_learning_rate.task_models[i,:,2]
            mean = int(human_vals.sum() / human_vals.shape[0])
            # print(mean)
            gaps = np.sqrt(self.human_learning_rate.task_models[i,:,1] ** 2 + self.human_learning_rate.task_models[i,:,3] ** 2).astype(int)
            gap = int(gaps.sum() / gaps.shape[0])
            # print(gap)
            lower = max(self.min_dur, mean - 3*gap)
            upper = min(mean + 3*gap, self.max_dur)
            for j in range(self.num_robots + self.num_humans):
                self.dur[i][j] = random.randint(lower, upper)
        # print(self.dur)
    
    def _generate_durations_graph(self):
        for i in range(self.num_tasks):
            # Add duration edges
            si = 's%03d' % (i+1)
            fi = 'f%03d' % (i+1)
            dur_min = self.dur[i].min().item() # convert from np.int32 to python int
            dur_max = self.dur[i].max().item()
            self.DG.add_weighted_edges_from([(si, fi, dur_max),
                                             (fi, si, -1 * dur_min)])
    
    # Absolute deadlines
    def _generate_ddl(self):
        deadline = round(self.num_tasks * 10 / self.num_robots)
            
        for i in range(1, self.num_tasks+1):
            if random.random() <= self.prob_deadline:
                dd = random.randint(1, deadline)                
                self.ddl.append([i, dd])
                
                fi = 'f%03d' % i
                self.DG.add_edge('s000', fi, weight = dd)
    
    # Wait constraints
    def _generate_wait(self):
        for i in range(1, self.num_tasks+1):
            for j in range(1, self.num_tasks+1):
                if i != j:
                    if random.random() <= self.prob_wait:
                        si = 's%03d' % i
                        fj = 'f%03d' % j
                        wait_time = random.randint(1,10)
                        # task i starts at least wait_time after task j finishes
                        self.DG.add_edge(si, fj, weight = -1 * wait_time)
                        
                        self.wait.append([i, j, wait_time])

    # Generate task locations
    # T x 2:  (x, y)
    def _generate_locs(self):
        self.locs = np.random.randint(1, self.map_width+1, 
                                      size = (self.num_tasks, 2))
        # TODO: release constraints
    # Only checks if the STN is consistent
    # Does not check if the STN + loc is consistenet
    
    def _check_consistency(self):
        updated = nx.floyd_warshall_numpy(self.DG).A
        consistent = True
        for i in range(updated.shape[0]):
            if updated[i, i] < 0:
                consistent = False
                break
        
        return consistent
    
    # Clear random constraints for re-generation
    def _reinitialize(self):
        self.DG.clear()
        self._initialize_graph()

    # Read From File
    def _read_from_file(self, fname, optimizer_read=False):
        """Read MRC Problem from a given file in the specific format

        Args:
            fname (string): fname that the MRC Problem is stored in
        Raises:
            FileNotFoundError: if given filename is not read, raises this error.
        """
        instance_file_name = fname + ".txt"
        print(instance_file_name)
        instance_file = open(instance_file_name, 'r')
        line_str = instance_file.readline()
        if line_str:
            split_data = line_str.split()
            self.num_tasks = int(split_data[0])
            self.num_robots = int(split_data[1])
            self.num_humans = int(split_data[2])
            self.map_width = int(split_data[3])
            self.min_dur = int(split_data[4])
            self.max_dur = int(split_data[5])
            self.prob_deadline = float(split_data[6])
            self.prob_wait = float(split_data[7]) / (self.num_tasks - 1)
            instance_file.close()
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), fname)
        
        # Read durations before initializing graph to set self.max_deadline with consistency
        self.dur = np.loadtxt(fname + '_dur.txt', dtype=int)
        self.human_learning_rate = HumanLearningRate(fname=fname + "_human.txt", num_tasks=self.num_tasks, num_robots=self.num_robots, num_humans=self.num_humans, noise = self.noise)
        self.set_max_makespan()

        self.DG = nx.DiGraph()

        self._initialize_graph()
        self._read_durations(fname)
        self._read_ddl(fname)
        self._read_wait(fname)
        self.locs = np.loadtxt(fname + '_loc.txt', dtype=int)

        if optimizer_read:
            # Optimizer
            self.sol = GModel(self.num_tasks, self.num_robots + self.num_humans, self.max_dur, fname + "_gurobi.log")
            self.sol.load_model(fname)
            # Optimal Solution
            if glob(fname + '_sol.txt'):
                # print("Found solution")
                np_schedule = np.loadtxt(fname + '_sol.txt', dtype=int)
                schedule_order = np.loadtxt(fname + '_sol_w.txt', dtype=int)
                if len(np_schedule.shape) == 1:
                    np_schedule = np.expand_dims(np_schedule, axis=1)
                self.optimal_schedule = ([], schedule_order.tolist())
                for row in np_schedule:
                    l = []
                    for x in row:
                        if x == 0:
                            break
                        l.append(x)
                    self.optimal_schedule[0].append(l)
                # print("Optimal:", self.optimal_schedule)

    def _read_durations(self, fname):
        for i in range(self.num_tasks):
            # Add duration edges
            si = 's%03d' % (i+1)
            fi = 'f%03d' % (i+1)
            dur_min = self.dur[i].min().item() # convert from np.int32 to python int
            dur_max = self.dur[i].max().item()
            self.DG.add_weighted_edges_from([(si, fi, dur_max),
                                             (fi, si, -1 * dur_min)])

    def _read_ddl(self, fname):
        np_ddl = np.loadtxt(fname + '_ddl.txt', dtype=int)
        if np_ddl.ndim == 1:
            np_ddl = np.expand_dims(np_ddl, axis=0)
        if np_ddl.shape[1] == 0:
            return
        self.ddl = np_ddl.tolist()
        for i, dd in self.ddl:
            fi = 'f%03d' % i
            self.DG.add_edge('s000', fi, weight = dd)
            
    def _read_wait(self, fname):
        np_wait = np.loadtxt(fname + '_wait.txt', dtype=int)
        if(np_wait.shape[0] == 0):
            return
        if np_wait.ndim == 1:
            # np_wait = np.expand_dims(np_wait, axis=0)
            self.wait = [np_wait.tolist()]
        else:
            self.wait = np_wait.tolist()
        for i, j, wait_time in self.wait:
            si = 's%03d' % i
            fj = 'f%03d' % j
            # task i starts at least wait_time after task j finishes
            self.DG.add_edge(si, fj, weight = -1 * wait_time)
            
    # Save To File
    def save_to_file(self, fname):
        """Saves MRC Problem to Folder

        Args:
            fname (string): name of the file to

        Returns:
            Boolean: True on success, False on failure
        """
        instance_file = open(fname + '.txt', 'w')
        # First Line
        prob_wait_ori = self.prob_wait * (self.num_tasks - 1)
        print(self.num_tasks, self.num_robots, self.num_humans, self.map_width, self.min_dur, self.max_dur, self.prob_deadline, prob_wait_ori, file=instance_file)
        instance_file.close()

        np.savetxt(fname+'_dur.txt', self.dur, fmt='%d')
        np_ddl = np.array(self.ddl)
        np.savetxt(fname+'_ddl.txt', np_ddl, fmt='%d')
        np_wait = np.array(self.wait)
        np.savetxt(fname+'_wait.txt', np_wait, fmt='%d')
        np.savetxt(fname+'_loc.txt', self.locs, fmt='%d')

        self.human_learning_rate.save_to_file(fname+'_human.txt')
        
        self.generate_solution(fname+'_gurobi.log')
        
        if self.sol is not None:
            self.sol.save_model(fname)
        if self.optimal_schedule is not None:
            np_order = np.array(self.optimal_schedule[1])
            np.savetxt(fname+'_sol_w.txt', np_order, fmt='%d')
            np_schedule = np.zeros([len(self.optimal_schedule[0]),len(max(self.optimal_schedule[0],key = lambda x: len(x)))])
            for i,j in enumerate(self.optimal_schedule[0]):
                np_schedule[i][0:len(j)] = j
            np.savetxt(fname+'_sol.txt', np_schedule, fmt='%d')
        return True

    def generate_solution(self, filename):
        try:
            schedule = self.get_optimal_with_gurobi(filename+'_gurobi.log', threads=4)
            return True
        except Exception as e:
            return False
            
    def get_max_makespan(self, task_indices):
        all_durations = np.array(self.dur)
        # print(all_durations, task_indices, all_durations.shape)
        durations = np.take(all_durations, task_indices, axis=0)
        # print(durations.sum(axis=0))
        max_deadline = durations.sum(axis=0).max() * self.max_deadline_multiplier
        return max_deadline

    def set_max_makespan(self):
        self.max_deadline = self.get_max_makespan([i for i in range(self.num_tasks)])

    def get_worst_worker(self, task_indices):
        all_durations = np.array(self.dur)
        # print(all_durations, task_indices, all_durations.shape)
        durations = np.take(all_durations, task_indices, axis=0)
        # print(durations.sum(axis=0))
        worker = durations.sum(axis=0).argmax()
        return worker
    
    # [Optional] Call Gurobi Solver to Get an Optimal Solution
    def get_optimal_with_gurobi(self, logfilename = '', bigM = 300, threads = 0):
        """Generates Optimal Schedule using gurobi

        Args:
            logfilename (str, optional): [description]. Defaults to ''.
            bigM (int, optional): [description]. Defaults to 300.
            threads (int, optional): [description]. Defaults to 0.
        """
        if self.sol == None:
            self.sol = GModel(self.num_tasks, self.num_robots + self.num_humans, self.max_dur, logfilename, bigM, threads)
        self.sol.add_temporal_cstr(self.dur, self.ddl, self.wait)
        self.sol.add_agent_constraints()
        # self.sol.add_loc_constraints(self.locs)
        self.sol.set_obj()
        self.sol.get_schedule()
        self.sol.optimize(timelimit=60)
        self.optimal = self.sol.optimal_solution()
        self.optimal_schedule = self.sol.get_schedule()
        # print("Gurobi Schedule:", self.optimal_schedule[0], self.optimal_schedule[1])
        return self.sol.optimal_solution(), self.sol.optimal_exists()
    
    # Only checks if the STN is consistent
    # Does not check if the STN + loc is consistenet
    def check_consistency(self):
        updated = nx.floyd_warshall_numpy(self.DG)
        consistent = True
        for i in range(updated.shape[0]):
            if updated[i, i] < 0:
                consistent = False
                break
        
        return consistent
    
    def initialize_STN(self):
        # Initialize directed graph    
        DG = nx.DiGraph()
        DG.add_nodes_from(['s000', 'f000'])
        DG.add_edge('s000', 'f000', weight = self.max_deadline)
                
        # Add task nodes
        for i in range(1, self.num_tasks+1):
            # Add si and fi
            si = 's%03d' % i
            fi = 'f%03d' % i
            DG.add_nodes_from([si, fi])
            DG.add_weighted_edges_from([(si, 's000', 0),
                                        ('f000', fi, 0)])
        
        # Add task durations
        for i in range(self.num_tasks):
            si = 's%03d' % (i+1)
            fi = 'f%03d' % (i+1)
            dur_min = self.dur[i].min().item()
            dur_max = self.dur[i].max().item()
            DG.add_weighted_edges_from([(si, fi, dur_max),
                                        (fi, si, -1 * dur_min)])
        
        # Add deadlines
        for i in range(len(self.ddl)):
            ti, ddl_cstr = self.ddl[i]
            fi = 'f%03d' % ti
            DG.add_edge('s000', fi, weight = ddl_cstr)            
            
        # Add wait constraints
        for i in range(len(self.wait)):
            ti, tj, wait_cstr = self.wait[i]
            si = 's%03d' % ti
            fj = 'f%03d' % tj
            DG.add_edge(si, fj, weight = -1 * wait_cstr)
        
        return DG
    
if __name__ == '__main__':
    # Testing  
    g = MRCProblem(num_tasks = 10, num_robots = 2, num_humans = 2)
    # g.generate_durations()
    # g.generate_ddl(prob_deadline=0.25)
    # g.generate_wait(prob_wait=0.25/19)
    # g.generate_locs()
    #g.save_data('ok')
    
    # print(sorted(g.DG.nodes))
    print(g.dur)
    # print(g.ddl)
    # print(g.wait)
    # print(g.locs)
    # print(g.DG.edges.data('weight'))
    # print(g.check_consistency())

    # # Consistency Check
    # g = MRCProblem(fname = "tmp/test_file")
    # print("Checking")
    # optimal, passed = g.get_optimal_with_gurobi("tmp/test_file_gurobi.log", threads=1)
    # print("Found Optimal Schedule Time:", optimal)

    # g.save_to_file("tmp/test_file")
    # # print(g.DG.edges())
    # h = MRCProblem(fname = "tmp/test_file")
    # # print(g.wait, h.wait)
    # # Make Sure that the read file and the stored file are consistent with each other.
    # assert(sorted(h.DG.nodes) == sorted(g.DG.nodes))
    # assert(np.all(g.dur == h.dur))
    # assert(np.all(g.ddl == h.ddl))
    # assert(np.all(g.wait == h.wait))
    # assert(np.all(g.locs == h.locs))
    # # print(h.DG.edges.data('weight'))
    # g_weights = sorted(list(g.DG.edges.data('weight')))
    # h_weights = sorted(list(h.DG.edges.data('weight')))
    # assert(sorted(g_weights) == sorted(h_weights))
    # assert(g.check_consistency() == h.check_consistency())

    
