# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:40:40 2020

@author: pheno

Calculate computation/run time
    only on sovled problems
"""

import os
import numpy as np
#import matplotlib.pyplot as plt
import networkx as nx


def time_summary(fname, return_mean_std = False):
    pre = np.load(fname+'.npy')[300:]
    pre_time = np.load(fname+'_time.npy')[300:]
    #print(pre[-1])
    
    a = np.full((1000,2), -1, dtype=np.float32)
    for i in range(len(pre)):
        idx = pre[i][0]
        a[round(idx-1)] = pre[i]
    
    a_t = np.full((1000,2), -1, dtype=np.float32)
    for i in range(len(pre_time)):
        idx = pre_time[i][0]
        a_t[round(idx-1)] = pre_time[i]
        
    total = a.shape[0]
    
    run_time = []
    for i in range(total):
        if a[i][1] > 0:
            run_time.append(a_t[i][1])
            
    time_np = np.array(run_time)
    print('feasible count:', len(time_np))
    
    if return_mean_std:
        time_mean = np.mean(time_np)
        time_std = np.std(time_np)
    
        return time_np, time_mean, time_std
    else:
        return time_np

def task_time(fname, folder):
    pre = np.load(fname+'.npy')
    pre_time = np.load(fname+'_time.npy')
    
    run_time = []
    run_task = []
    total = pre.shape[0]
    for i in range(total):
        dgfname = folder + '/%05d.gpickle' % (i+1)
        DG = nx.read_gpickle(dgfname)
        num_tasks = int(DG.number_of_nodes()/2 - 1)
        
        if pre[i][1] >= 0:
            run_time.append(pre_time[i])
            run_task.append(num_tasks)
    
    return np.array(run_task), np.array(run_time)

def Tercio_summary(fname):
    pre_time = np.loadtxt(fname)
    print('length', pre_time.shape)
    print('feasible', sum(pre_time>0))
    
    run_time = []
    for i in range(len(pre_time)):
        if pre_time[i] > 0:
            run_time.append(pre_time[i]/1000)
    
    time_np = np.array(run_time)
    
    return time_np

def time_100(folder, start_no = 1, end_no = 1000):
    run_time = []
    feas = 0
    
    for graph_no in range(start_no, end_no+1):        
        fname = folder + '/%05d.npy' % graph_no
        
        if os.path.isfile(fname):
            a = np.load(fname)
            
            if a[1] > 0:
                feas += 1
                fname_time = folder + '/%05d_time.npy' % graph_no
                a_t = np.load(fname_time)
                run_time.append(a_t[1])
    
    time_np = np.array(run_time)
    
    print(feas)
    
    return time_np
        
            
if __name__ == '__main__':
    '''
    mean20, std20 = time_summary('./r5/0901_fivetrain04_04000')
    print('Small problems',mean20,std20)
    
    mean50, std50 = time_summary('./r5/0727_fivetrain04_04000')
    print('Medium problems',mean50,std50)   

    mean100, std100 = time_summary('./r5/0811_fivetrain04_04000')
    print('Large problems',mean100,std100)    
    
    materials = ['Small', 'Medium', 'Large']
    x_pos = np.arange(len(materials))
    Means = [mean20, mean50, mean100]
    Stds = [std20, std50, std100]

    fig, ax = plt.subplots()
    ax.bar(x_pos, Means, yerr=Stds, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Running time for solving one instance ($s$)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(materials)
    ax.set_title('Running time of ScheduleNet on Different Problem Scales')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('running_time_bar.png')
    plt.show()
    '''
    '''
    # Plot whole dataset as points
    tasks20, times20 = task_time('./r5/0901_fivetrain04_04000', '../gen5/data0901')
    tasks50, times50 = task_time('./r5/0727_fivetrain04_04000', '../gen5/data0727')    
    tasks100, times100 = task_time('./r5/0811_fivetrain04_04000', '../gen5/data0811')    
    
    plt.figure(figsize=(10,7))
    plt.scatter(tasks20, times20)
    plt.scatter(tasks50, times50)
    plt.scatter(tasks100, times100)
    
    plt.grid(True)
    plt.legend(['Small', 'Medium', 'Large'],fontsize=20)
    plt.xlabel('Number of Tasks',fontsize=20)
    plt.ylabel('Running time for solving one instance ($s$)',fontsize=20)
    plt.title('Running time of ScheduleNet on Different Problem Scales',fontsize=20)
    
    #plt.show()
    plt.savefig('./run_time_scatter.png')
    '''
    time_np = time_summary('./johnsonU/r2t100_001_edf_v0')
    #time_np = Tercio_summary('./TercioResults/r10t100_001_myComputationTimes.csv')
    #time_np = time_100('./r/r10t1001v1', end_no = 1000)
    
    print('Mean:', np.mean(time_np))
    print('Std:', np.std(time_np))
    print('Median:', np.median(time_np))
    print('25%:', np.percentile(time_np, 25))
    print('75%:', np.percentile(time_np, 75))  