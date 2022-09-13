# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:22:34 2019

@author: pheno
Generate gurobi solutions’ makespan value
    
Two robot case
"""


import os
import sys
import networkx as nx
import numpy as np

sys.path.append('../')

from utils import SchedulingEnv

print('Gurobi results')

save_name = './r/r2t100_001_gurobi'
folder = '../gen/r2t100_001'
start_no = 1
end_no = 1000
total_no = end_no - start_no + 1
gurobi_count = 0
print(folder)

results = []

for graph_no in range(start_no, end_no+1):
    print('Evaluation on {}/{}'.format(graph_no, total_no))
    
    fname = folder + '/%05d' % graph_no
    
    # check if the graph is feasible for Gurobi
    solname = folder + 'v9/%05d' % graph_no
    solname_w = solname +'_w.txt'
    
    if os.path.isfile(solname_w):
        gurobi_count += 1
        
        env = SchedulingEnv(fname)
        
        optimals = []
        for i in range(env.num_robots):
            if os.path.isfile(solname+'_%d.txt' % i):
                optimals.append(np.loadtxt(solname+'_%d.txt' % i, dtype=np.int32))
            else:
                optimals.append([])
            
        optimalw = np.loadtxt(solname_w, dtype=np.int32) 
        
        for i in range(env.num_tasks):
            for j in range(env.num_robots):
                if optimalw[i] in optimals[j]:
                    rj = j
                    break           

            rt, reward, done = env.insert_robot(optimalw[i], rj)
        gurobi_opt = env.min_makespan
        print('Gurobi Feasible solution, min makespan: %f' % (gurobi_opt))
        results.append([graph_no, gurobi_opt])
    else:
        results.append([graph_no, -1])
        
# result summary
print('Gurobi feasible found: {}/{}'.format(gurobi_count, total_no))
# save results
results_np = np.array(results, dtype=np.float32)
np.save(save_name, results_np)