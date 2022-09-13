# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 17:27:01 2020

@author: pheno

Version: 2020-9-9

Gurobi model for solving the MRC problem
"""

from gurobipy import Model, Env, GRB, read, and_, LinExpr
import numpy as np

class GModel(object):

    def __init__(self, num_tasks = 20, num_workers = 5, max_dur = 100,
                 logfilename = '', bigM = 300, threads = 0):
        self.bigM = bigM
        self.num_tasks = num_tasks
        self.max_deadline = num_tasks * max_dur
        self.num_workers = num_workers
        
        env = Env(logfilename)
        env.setParam('OutputFlag', 0) # Suppress Print
        self.m = Model('tg', env=env)
        #self.m.Params.threads = threads 
        self.m.setParam('threads', threads) # these two are the same
        
        # start and end nodes
        self.m.addVar(ub=0.0, name='s000')
        self.m.addVar(name='f000')
        
        # node variable
        for i in range(1, self.num_tasks+1):
            si = 's%03d' % i
            fi = 'f%03d' % i
            self.m.addVar(name=si)
            self.m.addVar(name=fi)
            
        # binary decision variable - sequence
        # xij = 1 => task i occurs before task j
        # task i finishes before task j starts
        for i in range(1, self.num_tasks+1):
            for j in range(1, self.num_tasks+1):
                if i == j:
                    continue
                xij = 'x%03d%03d' % (i,j)
                self.m.addVar(vtype = GRB.BINARY, name = xij)
                
        # binary decision variable - assignment
        # Aai = 1 => task i assigned to robot a
        # in ten case, a = {0, 1, 2, 3, ..., 9}
        for a in range(self.num_workers):
            for i in range(1, self.num_tasks+1):
                Aai = 'A%03d%03d' % (a,i)
                self.m.addVar(vtype = GRB.BINARY, name = Aai)
        # Aaij = and(Aai, Aaj), i < j
        for a in range(self.num_workers):
            for i in range(1, self.num_tasks):
                for j in range(i+1, self.num_tasks+1):
                    Aaij = 'A%03d%03d%03d' % (a,i,j)
                    self.m.addVar(vtype = GRB.BINARY, name = Aaij)
            
        self.m.update()

    def set_obj(self):
        # Set objective
        f0 = self.m.getVarByName('f000')
        self.m.setObjective(f0, GRB.MINIMIZE)
        self.m.update()
    
    # dur, ddl, wait are numpy arrays (np.int32) with
    # corresponding sizes
    def add_temporal_cstr(self, dur, ddl, wait):
        # add max deadline
        #s0 = self.m.getVarByName('s000')
        f0 = self.m.getVarByName('f000')
        self.m.addConstr(f0 <= self.max_deadline)
        
        # add duration constraints
        # plus si >= s0
        # plus fi <= f0
        for i in range(1, self.num_tasks+1):
            si = 's%03d' % i
            fi = 'f%03d' % i
            si_ = self.m.getVarByName(si)
            fi_ = self.m.getVarByName(fi)
            # general constraints
            #self.m.addConstr(si_ - s0 >= 0) this is implied by definition
            self.m.addConstr(fi_ - f0 <= 0)
            
            for a in range(self.num_workers):
                duration = dur[i-1][a].item()
                Aai = 'A%03d%03d' % (a,i)
                Aai_ = self.m.getVarByName(Aai)

                # use indicator constraints
                self.m.addConstr((Aai_ == 1) >> (fi_ - si_ == duration))

        # add deadline constraints
        # fi <= ddl_cstr
        for i in range(len(ddl)):
            ti, ddl_cstr = ddl[i]
            fi = 'f%03d' % ti
            fi_ = self.m.getVarByName(fi)
            self.m.addConstr(fi_ <= ddl_cstr)
        
        # add wait constraints
        # si >= fj + wait_cstr
        # si - fj >= wait_cstr
        for i in range(len(wait)):
            ti, tj, wait_cstr = wait[i]
            si = 's%03d' % ti
            fj = 'f%03d' % tj
            si_ = self.m.getVarByName(si)
            fj_ = self.m.getVarByName(fj)
            self.m.addConstr(si_ - fj_ >= wait_cstr)

    def add_agent_constraints(self):
        # Ensure each task is assigned to one robot
        for i in range(1, self.num_tasks+1):
            tmp_str = 'A%03d%03d' % (0, i)
            lin_expr = LinExpr(self.m.getVarByName(tmp_str))
            for j in range(1, self.num_workers):
                tmp_str = 'A%03d%03d' % (j, i)
                tmp_var = self.m.getVarByName(tmp_str)
                lin_expr.addTerms(1.0, tmp_var)
            self.m.addConstr(lin_expr == 1)
        
        # Tasks assigned to the same robot don't overlap
        # Each robot only performs one task at a time.
        # Xij + Xji == 1 constraints if Aai = Aaj = 1
        # that is, Aaij = and (Aai, Aaj) = 1
        for a in range(self.num_workers):
            for i in range(1, self.num_tasks):
                for j in range(i+1, self.num_tasks+1):
                    xij = 'x%03d%03d' % (i,j)
                    xji = 'x%03d%03d' % (j,i)
                    Aai = 'A%03d%03d' % (a,i)
                    Aaj = 'A%03d%03d' % (a,j)
                    Aaij = 'A%03d%03d%03d' % (a,i,j)
                    xij_ = self.m.getVarByName(xij)
                    xji_ = self.m.getVarByName(xji)
                    Aai_ = self.m.getVarByName(Aai)
                    Aaj_ = self.m.getVarByName(Aaj)
                    Aaij_ = self.m.getVarByName(Aaij)
                    self.m.addConstr(Aaij_ == and_(Aai_, Aaj_))
                    self.m.addConstr((Aaij_ == 1) >> (xij_ + xji_ == 1))
        
        # What happens for Xij == 1, use indicator constraints
        for i in range(1, self.num_tasks+1):
            for j in range(1, self.num_tasks+1):
                if i == j:
                    continue
                xij = 'x%03d%03d' % (i,j)
                fi = 'f%03d' % i
                sj = 's%03d' % j
                xij_ = self.m.getVarByName(xij)
                fi_ = self.m.getVarByName(fi)
                sj_ = self.m.getVarByName(sj)
                self.m.addConstr((xij_ == 1) >> (fi_ - sj_ <= 0))
                # use bigM
                #self.m.addConstr(sj_ - fi_ >= self.bigM * (xij_ - 1))
        
        self.m.update()

    def add_loc_constraints(self, locs, diff = 1.0):
        # Tasks within certain range don’t overlap
        # For all pairs of i and j
        # if |loc_i - loc_j| <= r, then Xij + Xji == 1
        for i in range(1, self.num_tasks):
            xi, yi = locs[i-1]
            for j in range(i+1, self.num_tasks+1):
                xj, yj = locs[j-1]
                dist_2 = (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj)
                if dist_2 <= diff * diff:
                    xij = 'x%03d%03d' % (i,j)
                    xji = 'x%03d%03d' % (j,i)                    
                    xij_ = self.m.getVarByName(xij)
                    xji_ = self.m.getVarByName(xji)                    
                    self.m.addConstr(xij_ + xji_ == 1)                    

        self.m.update()
    
    def save_model(self, file_name):
        self.m.write(file_name+'.mps')
    
    def load_model(self, file_name):
        self.m = read(file_name+'.mps', env=Env())
    
    def save_solution(self, file_name):
        if self.m.solCount > 0:
            self.m.write(file_name+'.sol')
            return True
        else:
            return False    

    def optimize(self, timelimit=60*60):
        #self.m.getParamInfo('TimeLimit')
        self.m.setParam('TimeLimit', timelimit)
        self.m.optimize()
    
    def show_status(self):
        if self.m.status == GRB.Status.OPTIMAL:
            print('Optimal objective: %g' % self.m.objVal)
        elif self.m.status == GRB.Status.INF_OR_UNBD:
            print('Model is infeasible or unbounded')
        elif self.m.status == GRB.Status.INFEASIBLE:
            print('Model is infeasible')
        elif self.m.status == GRB.Status.UNBOUNDED:
            print('Model is unbounded')
        else:
            print('Optimization ended with status %d' % self.m.status)        

    def optimal_solution(self):
        return self.m.objVal
        
    def optimal_exists(self):
        return self.m.status == GRB.Status.OPTIMAL

    def get_schedule(self):
        if self.m.solCount == 0:
            # print('No solution available')
            return None
        
        # get task assignemnts of each robot 
        assignment = [[] for i in range(self.num_workers)]
        for a in range(self.num_workers):
            for i in range(1, self.num_tasks+1):
                Aai = 'A%03d%03d' % (a,i)
                Aai_ = self.m.getVarByName(Aai)
                Aai_int = round(Aai_.x)
                if Aai_int == 1:
                    assignment[a].append(i)        
        
        # for each robot, get schedule from assignment
        # sum up decision variables xij for each i
        schedule = [[] for i in range(self.num_workers)]
        for a in range(self.num_workers):
            cnt = np.zeros(len(assignment[a]), dtype=float)
            for i in range(len(assignment[a])):
                for j in range(len(assignment[a])):
                    if i == j:
                        continue
                    ti = assignment[a][i]
                    tj = assignment[a][j]
                    xij = 'x%03d%03d' % (ti,tj)
                    xij_ = self.m.getVarByName(xij)
                    cnt[i] += xij_.x
            # ascending of negative cnt = descending of cnt
            tmp = np.argsort(-cnt)
            # get actual schedule
            for i in range(len(tmp)):
                schedule[a].append(assignment[a][tmp[i]])
                
        # get the order of all tasks using task start time
        task_start_time = np.zeros(self.num_tasks, dtype=float)
        for i in range(1, self.num_tasks+1):
            si = 's%03d' % i
            si_ = self.m.getVarByName(si)
            task_start_time[i-1] = si_.x
        whole_schedule = np.argsort(task_start_time) + 1
        
        return schedule, whole_schedule


if __name__ == '__main__':
    # Testing  
    gm = GModel(2, threads=2)
    gm.set_obj()
    gm.get_schedule()
    print(gm.m)
    print(len(gm.m.getVars()))
