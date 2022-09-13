# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 12:01:03 2021

@author: pheno

Quick load gurobi results from xx.sol
"""


import os

import numpy as np

print('Gurobi results')

save_name = './r/r10t200_001_gurobi'
folder = '../gen/r10t200_001'
start_no = 1
end_no = 100
total_no = end_no - start_no + 1
gurobi_count = 0
print(folder)

results = []

for graph_no in range(start_no, end_no+1):
    print('Loading on {}/{}'.format(graph_no, total_no))
    
    solname_w = folder + 'v9/%05d.sol' % graph_no
    
    if os.path.isfile(solname_w):
        gurobi_count += 1
        with open(solname_w) as f:
            line = f.readline()
            obj = float(f.readline().replace('# Objective value = ', ''))
            results.append([graph_no, round(obj)])
    else:
        results.append([graph_no, -1])

# result summary
print('Gurobi feasible found: {}/{}'.format(gurobi_count, total_no))
# save results
results_np = np.array(results, dtype=np.float32)
np.save(save_name, results_np)
