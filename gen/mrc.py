# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 19:11:35 2020

@author: pheno

version: 2020-9-9

Generate one problem instance
    also check the STN consistency
"""


import random
import networkx as nx
import numpy as np

class MRCProblem(object):
    def __init__(self, num_tasks = 20, num_robots = 5, map_width = 5):
        self.num_tasks = num_tasks
        self.max_deadline = num_tasks * 10
        
        self.num_robots = num_robots
        
        self.map_width = map_width
        
        self.DG = nx.DiGraph()
        
        self.initialize()

    def initialize(self):
        # Constraints
        self.dur = np.zeros((self.num_tasks, self.num_robots), dtype=np.int32)
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
    # Task durations
    def generate_durations(self):
        for i in range(self.num_tasks):
            mean = random.randint(1, 10)
            gap = random.randint(1, 3)
            lower = max(1, mean - gap)
            upper = min(mean + gap, 10)
            for j in range(self.num_robots):
                self.dur[i][j] = random.randint(lower, upper)
            # Add duration edges
            si = 's%03d' % (i+1)
            fi = 'f%03d' % (i+1)
            dur_min = self.dur[i].min().item() # convert from np.int32 to python int
            dur_max = self.dur[i].max().item()
            self.DG.add_weighted_edges_from([(si, fi, dur_max),
                                             (fi, si, -1 * dur_min)])
    # Absolute deadlines
    def generate_ddl(self, prob_deadline):
        deadline = round(self.num_tasks * 10 / self.num_robots)
            
        for i in range(1, self.num_tasks+1):
            if random.random() <= prob_deadline:
                dd = random.randint(1, deadline)                
                self.ddl.append([i, dd])
                
                fi = 'f%03d' % i
                self.DG.add_edge('s000', fi, weight = dd)
    
    # Wait constraints
    def generate_wait(self, prob_wait):
        for i in range(1, self.num_tasks+1):
            for j in range(1, self.num_tasks+1):
                if i != j:
                    if random.random() <= prob_wait:
                        si = 's%03d' % i
                        fj = 'f%03d' % j
                        wait_time = random.randint(1,10)
                        # task i starts at least wait_time after task j finishes
                        self.DG.add_edge(si, fj, weight = -1 * wait_time)
                        
                        self.wait.append([i, j, wait_time])

    # Generate task locations
    # T x 2:  (x, y)
    def generate_locs(self):
        self.locs = np.random.randint(1, self.map_width+1, 
                                      size = (self.num_tasks, 2))

    # All-in-one
    def generate_constraints(self, prob_deadline, prob_wait):
        self.generate_durations()
        self.generate_ddl(prob_deadline)
        self.generate_wait(prob_wait)
        self.generate_locs()
        
    # Only checks if the STN is consistent
    # Does not check if the STN + loc is consistenet
    def check_consistency(self):
        updated = nx.floyd_warshall_numpy(self.DG).A
        consistent = True
        for i in range(updated.shape[0]):
            if updated[i, i] < 0:
                consistent = False
                break
        
        return consistent
    
    # Clear random constraints for re-generation
    def reinitialize(self):
        self.DG.clear()
        self.initialize()

    def save_data(self, fname):
        np.savetxt(fname+'_dur.txt', self.dur, fmt='%d')
        np_ddl = np.array(self.ddl)
        np.savetxt(fname+'_ddl.txt', np_ddl, fmt='%d')
        np_wait = np.array(self.wait)
        np.savetxt(fname+'_wait.txt', np_wait, fmt='%d')
        np.savetxt(fname+'_loc.txt', self.locs, fmt='%d')
        
    
if __name__ == '__main__':
    # Testing  
    g = MRCProblem(num_tasks = 16, num_robots = 5, map_width = 4)
    # g.generate_durations()
    # g.generate_ddl(prob_deadline=0.25)
    # g.generate_wait(prob_wait=0.25/19)
    # g.generate_locs()
    g.generate_constraints(prob_deadline=0.25, prob_wait=0.25/19)
    #g.save_data('ok')
    
    print(sorted(g.DG.nodes))
    print(g.dur)
    print(g.ddl)
    print(g.wait)
    print(g.locs)
    print(g.DG.edges.data('weight'))
    print(g.check_consistency())
