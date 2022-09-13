# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:41:10 2019

@author: pheno
Baselines - earlist deadline first
    compare with gurobi quantity and quality
    v0 robots are being picked in their idx order
"""


import sys
import numpy as np
import time

from edfutils import RobotTeam, pick_task

sys.path.append('../')
from utils import SchedulingEnv

print('Baseline: earlist deadline first')

save_name = './johnsonU/r10t100_001_edf_v0'
folder = '../gen/r10t100_001'
start_no = 1
end_no = 1000
total_no = end_no - start_no + 1
feas_count = 0
print(folder)

results = []
record_time = []

for graph_no in range(start_no, end_no+1):
    print('Evaluation on {}/{}'.format(graph_no, total_no))
    start_t = time.time()
    
    fname = folder + '/%05d' % graph_no
    env = SchedulingEnv(fname)
    
    robots = RobotTeam(env.num_robots)
    
    terminate = False
    
    for t in range(env.num_tasks * 10):
        robot_available = robots.pick_robot(t)
        
        # using for loop means that the robots are picked in their idx order
        for robot_chosen in robot_available:            
            valid_task = env.get_valid_tasks(t)
            if len(valid_task) > 0:
                # get an updated version of STN w.r.t the chosen robot and valid tasks
                min_rSTN, consistent = env.get_rSTN(robot_chosen, valid_task)
                
                if consistent:
                    task_chosen = pick_task(min_rSTN, valid_task, t)
                else:
                    task_chosen = -1
                
                if task_chosen >= 0:
                    task_dur = env.dur[task_chosen-1][robot_chosen]
                    rt, reward, done = env.insert_robot(task_chosen, robot_chosen)
                    robots.update_status(task_chosen, robot_chosen, task_dur, t)
                    #print(task_chosen, robot_chosen, task_dur)
                    '''
                    Check for termination
                    '''
                    if rt == False:
                        print('Infeasible after %d insertions' % (len(env.partialw)-1))
                        results.append([graph_no, -1])
                        terminate = True
                        break
                    elif env.partialw.shape[0]==(env.num_tasks+1):
                        feas_count += 1
                        edf_opt = env.min_makespan
                        print('Feasible solution found, min makespan: %f' 
                              % (env.min_makespan))
                        results.append([graph_no, edf_opt])
                        terminate = True
                        break
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
    np.save(save_name, results_np)
    
    # save computation time
    record_time_np = np.array(record_time, dtype=np.float32)
    np.save(save_name+'_time', record_time_np)

# result summary
print('Feasible solution found: {}/{}'.format(feas_count, total_no))
print('Average time per instance:  {:.4f}'.format(sum(record_time_np[:,1])/total_no))