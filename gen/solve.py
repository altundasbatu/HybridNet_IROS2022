# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 19:35:06 2020

@author: pheno

Version: 2020-9-9

Load problems and solve using Gurobi v9
"""

from gmodel import GModel
import numpy as np

def load_and_solve(time_limit, fname, savename, diff, threads):
    '''
    Step 1. Load problem instance
    '''
    dur = np.loadtxt(fname+'_dur.txt', dtype=np.int32)    
    ddl = np.loadtxt(fname+'_ddl.txt', dtype=np.int32)
    wait = np.loadtxt(fname+'_wait.txt', dtype=np.int32)
    loc = np.loadtxt(fname+'_loc.txt', dtype=np.int32)    
    
    num_tasks = dur.shape[0]
    num_robots = dur.shape[1]
    
    # reshape if shape is one-dimension, meaning there is only one constraint
    if len(ddl) > 0 and len(ddl.shape) == 1:
        ddl = ddl.reshape(1, -1)

    if len(wait) > 0 and len(wait.shape) == 1:
        wait = wait.reshape(1, -1)
    
    '''
    Step 2. Pass the graph to gurobi and solve for optimal schedule
    '''
    gm = GModel(num_tasks, num_robots, savename + '.log', threads = threads)
    gm.set_obj()
    
    # Temporal constraints
    gm.add_temporal_cstr(dur, ddl, wait)
    
    # Same agent constraints
    gm.add_agent_constraints()
    
    # Near location constraints
    gm.add_loc_constraints(locs = loc, diff = diff)
    
    # Optimize
    gm.optimize(time_limit)
    gm.show_status()

    '''
    Step 3. Save & Parse Results
    '''
    # Save gurobi model
    gm.save_model(savename)
    print('Gurobi model saved.')
    # Save gurobi solution
    ret = gm.save_solution(savename)
    if ret:
        print('Gurobi solution saved.')
        # Get schedule sequence from the solution
        schedule, whole_schedule = gm.get_schedule()
        for i in range(num_robots):
            np.savetxt(savename+'_%d.txt'%i, schedule[i], fmt='%d')

        np.savetxt(savename+'_w.txt', whole_schedule, fmt='%d')
        print('Schedule saved.')
    else:
        print('No solution found.')

if __name__ == '__main__':
    
    folder = './r10t20_002'
    start_number = 62
    end_number = 400
    
    time_limit = 300 
    diff = 1.0
    threads = 10
    
    for i in range(start_number, end_number+1):
        fname = folder + '/%05d' % i
        savename = folder + 'v9/%05d' % i
        #savename = 'tmp2' + '/%05d' % i
        print('Generating results for %05d' % i)
        #print(num_tasks)
        load_and_solve(time_limit, fname, savename, diff, threads)
