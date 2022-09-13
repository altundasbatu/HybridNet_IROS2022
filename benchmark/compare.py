# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:21:59 2019

Compare solution quality of different methods

@author: pheno

input format
    1   xxx
    2   xxx
    3   xxx
    ...
    100/1000 xxx
    
    xxx: 
        positive valuee for feaisble solution
        -1 for infeasible, 
"""

import numpy as np

# compare solution quality use truth as optimal
def compare(truth, pre):
    total_num = truth.shape[0]
    feas_count = 0
    truth_count = 0
    ratio = 0.0
    missed = 0
    details = []
    
    for i in range(total_num):
        truth_opt = truth[i][1]
        pre_opt = pre[i][1]
        
        if truth_opt > 0:
            truth_count += 1
        
        if pre_opt > 0:
            feas_count += 1
            if truth_opt > 0:
                ratio += pre_opt / truth_opt
                details.append([pre[i][0], pre_opt / truth_opt])
            else:
                missed += 1
                #print('Great, pre finds a solution where truth fails! %d'
                #     % truth[i][0])
    
    # result summary
    print('Pre: {}/{}'.format(feas_count, total_num))
    # compare with gurobi results
    print('Truth: {}/{}'.format(truth_count, total_num))
    if feas_count - missed > 0:
        print('Opt ratio:', ratio / (feas_count - missed)) 
    
    return details

# compare solution quality use truth as optimal, with mask
def compare_with_mask(truth, pre, mask):
    total_num = truth.shape[0]
    mask_count = 0
    truth_count = 0
    ratio = 0.0
    missed = 0
    
    for i in range(total_num):
        truth_opt = truth[i][1]
        pre_opt = pre[i][1]
        mask_opt = mask[i][1]
                
        if truth_opt >= 0:
            truth_count += 1
            
        if mask_opt >= 0:
            mask_count += 1
            if truth_opt >= 0:
                if pre_opt >= 0:
                    ratio += pre_opt / truth_opt
                else:
                    missed += 1
            else:
                print('Great, mask finds a solution where truth fails!') 
    
    # result summary
    print('Pre with mask: {}({})/{}'.format(mask_count, missed, total_num))
    # compare with gurobi results
    print('Truth: {}/{}'.format(truth_count, total_num))
    if mask_count - missed > 0:
        print('Opt ratio:', ratio / (mask_count - missed))    

# pick the best for each problem from pre_list as the final results
def compare_ensemble(truth, pre_list):
    total_num = truth.shape[0]
    feas_count = 0
    truth_count = 0
    ratio = 0.0
    missed = 0
    
    '''
    combine results
    '''
    # num_list x total_num
    tmp = np.array([value[:, 1] for value in pre_list])
    # replace -1 with larger number and get min
    tmp[tmp < 0] = 1000.0
    pre = np.amin(tmp, axis = 0)
    # replace larger number with -1
    pre[pre > 999.0] = -1.0
    
    for i in range(total_num):
        truth_opt = truth[i][1]
        pre_opt = pre[i]
        
        if truth_opt >= 0:
            truth_count += 1
        
        if pre_opt >= 0:
            feas_count += 1
            if truth_opt >= 0:
                ratio += pre_opt / truth_opt
            else:
                missed += 1
                #print('Great, pre finds a solution where truth fails!')
    
    # result summary
    print('Pre: {}/{}'.format(feas_count, total_num))
    # compare with gurobi results
    print('Truth: {}/{}'.format(truth_count, total_num))
    if feas_count - missed > 0:
        print('Co-found:', feas_count - missed)
        print('Opt ratio:', ratio / (feas_count - missed)) 
        
    return pre
    

if __name__ == '__main__':
    '''
    truth_name = './r/r10t200_001_gurobi'
    pre_name = './johnsonU/v1_r10t200_001_checkpoint_01500'
    #pre_name = './Merged/r10t200_001_merged'
    #pre_name = './r/r10t20_001_Tercio'
    #mask_name = './r10/1205_sltrain05_18000'
    
    truth = np.load(truth_name+'.npy')
    # truth_1000 = np.load(truth_name+'.npy')
    # truth = truth_1000[:500]
    
    pre = np.load(pre_name+'.npy')#[:170]
    print(pre[-1])
    print(len(pre))
    
    a = np.full(truth.shape, -1, truth.dtype)
    for i in range(len(pre)):
        idx = pre[i][0]
        a[round(idx-1)] = pre[i]
    
    # For Tercio
    # a = np.full(truth.shape, -1, truth.dtype)
    # t = np.loadtxt(pre_name+'.txt', dtype = truth.dtype)
    # a[:,0] = list(range(1,1001))
    # a[:200,1] = t[:200]

    details = compare(truth, a)
    #print(details)
    
    #print('Using mask from EDF')
    #mask = np.load(mask_name+'.npy')
    #compare_with_mask(truth, pre, mask)
    '''
    
    truth_name = './r/r10t100_001_gurobi'
    truth = np.load(truth_name+'.npy')
    
    name_list = ['./johnsonU/r10t100_001_edf_v0',
                 './johnsonOld/r10t100_001_edf_v1']

    pre_list = []
    for pre_name in name_list:
        pre = np.load(pre_name+'.npy')
        a = np.full(truth.shape, -1, truth.dtype)
        for i in range(len(pre)):
            idx = pre[i][0]
            a[round(idx-1)] = pre[i]
        pre_list.append(a)
        
    print('Results of each single model:')
    for pre in pre_list:
        compare(truth, pre)
    
    print('Model ensemble results:')
    results = compare_ensemble(truth, pre_list)
    
    print(results.shape)
    ensemble = np.copy(truth)
    ensemble[:,1] = results
    np.save('./johnsonU/r10t100_001_edf_comb', ensemble)
    