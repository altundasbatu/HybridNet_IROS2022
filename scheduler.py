# -*- coding: utf-8 -*-
"""
Created on Fri October 14 12:47:42 2021

@author: baltundas

"""

import torch

import torch.nn as nn
import numpy as np

import copy

import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from env.mrc_problem import MRCProblem
from env.hybrid_team import HybridTeam

from env.scheduling_env import SchedulingEnv
from env.multi_round_scheduling_env import MultiRoundSchedulingEnv

from utils import ReplayMemory, Transition, action_helper_rollout
from utils import hetgraph_node_helper, build_hetgraph

from evolutionary_algorithm import random_gen_schedules, swap_task_allocation, generate_evolution

from policy_net import HybridPolicyNetUpdateSelected, HybridPolicyNetNonRecursive

class PGScheduler():
    """ Policy Gradient Scheduler basex on Repair21

    """
    def __init__(self, device = torch.device("cpu"), 
                 nn = 'hybridnet',
                 gamma=0.95, lr=1e-4,
                 weight_decay=1e-5, lmbda = 0.95,
                 milestones=[30, 80], lr_gamma=0.1, 
                 entropy_coefficient=0.0, 
                 selection_mode='sample',
                 verbose='none'):
        self.device = device
        # self.model = HybridPolicyNet(selection_mode=selection_mode) #.to(self.device)
        # self.model = HybridPolicyNetSparse(selection_mode=selection_mode) #.to(self.device)
        self.model = HybridPolicyNetUpdateSelected(selection_mode=selection_mode, verbose=verbose, device=device).to(self.device)
        # self.model = HybridPolicyNetUpdateSelectedNonBias(selection_mode=selection_mode) #.to(self.device)
        if nn == 'hetgat':
            self.model = HybridPolicyNetNonRecursive(selection_mode=selection_mode, verbose=verbose, device=device).to(self.device)
        
        self.saved_log_probs = []
        self.rewards = []
        self.gamma = gamma
        self.lmbda = lmbda
        self.lr = lr
        self.weight_decay = weight_decay
        self.entropy_coefficient = entropy_coefficient

        self.eps = np.finfo(np.float32).eps.item()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones = milestones,
                                        gamma = lr_gamma)
    
    def load_checkpoint(self, trained_checkpoint, retain_old: bool = True):
        cp = torch.load(trained_checkpoint)
        self.model.load_state_dict(cp['policy_net_state_dict'])
        self.optimizer.load_state_dict(cp['optimizer_state_dict'])
        if retain_old:
            self.lr_scheduler.load_state_dict(cp['scheduler_state_dict'])
        else:
            # print("Optimizer", self.optimizer.state_dict['lr'], cp['optimizer_state_dict'])
            # Scheduler Import relevant parts of the scheduler:
            relevant_lr_scheduler_state_dict = self.lr_scheduler.state_dict()
            relevant_lr_scheduler_state_dict['last_epoch'] = cp['scheduler_state_dict']['last_epoch']
            relevant_lr_scheduler_state_dict['_step_count'] = cp['scheduler_state_dict']['_step_count']
            self.lr_scheduler.load_state_dict(relevant_lr_scheduler_state_dict)
            print(self.lr_scheduler.state_dict())
        return cp['i_batch'] + 1

    def get_variables(self, env):
        # Unscheduled Tasks
        num_tasks = env.problem.num_tasks
        num_robots = env.team.num_robots
        num_humans = env.team.num_humans
        curr_g = copy.deepcopy(env.halfDG)
        curr_partials = copy.deepcopy(env.partials)
        curr_partialw = copy.deepcopy(env.partialw)
        durs = copy.deepcopy(env.dur)
        # Act Robot is not used for this model of Scheduler
        act_robot = 0
        unsch_tasks = np.array(action_helper_rollout(num_tasks, curr_partialw), dtype=np.int64)
        # Graph Neural Network
        g = build_hetgraph(curr_g,
                            num_tasks, num_robots, num_humans,
                            durs,
                            curr_partials, unsch_tasks)
        # Feature Dictionary
        num_actions = len(unsch_tasks)
        feat_dict = hetgraph_node_helper(curr_g.number_of_nodes(), 
                                         curr_partialw, 
                                         curr_partials,
                                         # transition.locs, 
                                         durs, 
                                         # map_width, 
                                         num_robots + num_humans, num_actions)
        
        feat_dict_tensor = {}
        for key in feat_dict:
            feat_dict_tensor[key] = torch.Tensor(feat_dict[key]).to(device)
        
        return g, feat_dict_tensor, unsch_tasks

    def select_action(self, env, genetic=False):
        """Generate a Schedule as Action for a MultiRoundEnvironment
        Args:
            env (SingleRoundScheduler): Single-Round Scheduler Environment
        """
        # No Grad
        with torch.no_grad():
            schedule = self.model(env)
        if genetic:
            schedule = self.run_genetic(schedule, env)
        return schedule
    
    def run_genetic(self, schedule, env, generation=10, base_population = 90, new_mutation=10, new_random=10):
        # if there are unscheduled_tasks, add them to the schedule with worst worker in total
        unscheduled_tasks = env.get_unscheduled_tasks()
        # print(schedule, env.min_makespan)
        # print("Unscheduled:", unscheduled_tasks)
        # if infeasible, complete the schedule
        if len(unscheduled_tasks) > 0:
            worker = env.problem.get_worst_worker(unscheduled_tasks-1)
            for u_task in unscheduled_tasks:
                schedule.append([u_task, worker, 1.0])
        # print(schedule, env.min_makespan)
        # Run Evolutionary Algorithm to generate more schedules:
        env.reset() # Reset the Single Round Environment
        new_random_schedules = random_gen_schedules([schedule], env.team, base_population + new_random - 1)
        if len(new_random_schedules) == 0: # if there can be no swaps made
            new_random_schedules = generate_evolution([schedule], base_population + new_random - 1, [0])
        new_mutations = generate_evolution([schedule], new_mutation, [0])
        new_generation = [schedule] + new_random_schedules + new_mutations # include the baseline
        # Store the previously generated scores
        scores = [[], []] # Scores for Infeasible and Feasible Schedule Scores
        schedules = [[], []] # Schedules for Infeasible and Feasible Schedule Scores
        # multi_round_env = MultiRoundSchedulingEnv(env.problem, env.team)
        infeasible_idx_sorted = []
        feasible_idx_sorted = []
        for gen in range(generation):
            # print(scores[1])
            # indices = [[], []] # Indicies for Infeasible and Feasible Schedules
            # Run MultiRound to get total score/feasibility for the new generation
            for j in range(len(new_generation)):
                env.reset()
                # print(schedule_i)
                schedule_i = new_generation[j]
                # print(schedule_i)
                rt = False
                for step in schedule_i:
                    # print(step)
                    rt, reward, done, _ = env.step(step[0], step[1], step[2])
                    if rt == False: # Infeasible
                        scores[0].append(env.problem.max_deadline)
                        schedules[0].append(schedule_i)
                        break
                if rt:
                    scores[1].append(env.min_makespan)
                    schedules[1].append(schedule_i)
                # if j == 0: print(scores)
            # Select the top evolution_cutoff indices for next generation
            # print(indices, scores)
            if len(scores[0]) != 0:
                infeasible_idx_sorted = np.argsort(scores[0])
            if len(scores[1]) != 0:
                feasible_idx_sorted = np.argsort(scores[1])
            # print(infeasible_idx_sorted, feasible_idx_sorted)                
            
            if gen < generation - 1: # For all but last step, take the top base_population of the scores and schedules
                feasible_top_n = min(base_population, len(scores[1]))
                infeasible_top_n = max(0, base_population - len(scores[1]))
                # Get the top elements of the array
                schedules[1] = np.array(schedules[1], dtype=int)[feasible_idx_sorted][:feasible_top_n].tolist() 
                scores[1] = np.array(scores[1], dtype=int)[feasible_idx_sorted][:feasible_top_n].tolist()
                if infeasible_top_n > 0:
                    schedules[0] = np.array(schedules[0], dtype=int)[infeasible_idx_sorted][:infeasible_top_n].tolist()
                    scores[0] = np.array(scores[0], dtype=int)[infeasible_idx_sorted][:infeasible_top_n].tolist()
                else:
                    schedules[0] = []
                    scores[0] = []
                # print(schedules[0] + schedules[1])
                # Generate new_random number of random schedules
                random_schedules = random_gen_schedules(schedules[0] + schedules[1], env.team, new_random)
                if len(random_schedules) == 0:
                    random_schedules = swap_task_allocation(schedules[0] + schedules[1], new_random)
                # Generate new_mutation number of mutation schedules
                new_mutation_schedules = swap_task_allocation(schedules[0] + schedules[1], new_mutation)
                if len(new_mutation_schedules) == 0:
                    new_mutation_schedules = random_gen_schedules(schedules[0] + schedules[1], env.team, new_mutation)
                new_generation = random_schedules + new_mutation_schedules
                # update schedules to baseline
                # schedules = baselines
                
        if len(feasible_idx_sorted) != 0: # there is a feasible solution
            return schedules[1][0]
        else: # the solution is infeasible
            return schedules[0][0]
        return []
    
    def initialize_batch(self, batch_size):
        self.batch_saved_t_log_probs = [[] for i in range(batch_size)]
        self.batch_saved_t_entropy = [[] for i in range(batch_size)]
        self.batch_saved_w_log_probs = [[] for i in range(batch_size)]
        self.batch_saved_w_entropy = [[] for i in range(batch_size)]
        self.batch_rewards = [[] for i in range(batch_size)]

    def batch_select_action(self, env, i_b):
        """Batch Selection of Action

        Args:
            env
        """
        
        # Reset the model log_prob buffer to save space.
        ## Task Classifier
        self.model.task_classifier.saved_log_probs = []
        self.model.task_classifier.saved_entropy = []
        ## Worker Classifier
        self.model.worker_classifier.saved_log_probs = []
        self.model.worker_classifier.saved_entropy = []
        
        # produced variables required for the model
        # Generate a schedule
        schedule = self.model(env)
        # Add the log probabilities to the batch data
        self.batch_saved_t_log_probs[i_b].append(self.model.task_classifier.saved_log_probs[-1])
        self.batch_saved_w_log_probs[i_b].append(self.model.worker_classifier.saved_log_probs[-1])
        # Add the entropy to the batch data
        self.batch_saved_t_entropy[i_b].append(self.model.task_classifier.saved_entropy[-1])
        self.batch_saved_w_entropy[i_b].append(self.model.worker_classifier.saved_entropy[-1])
        
        return schedule

    def batch_finish_episode(self, batch_size, num_rounds = 1, max_norm=0.75):        
        '''
        Batch version
        '''
        batch_policy_loss = [[] for i in range(batch_size)]
        batch_total_loss = []
        
        # zero-pad episodes with early termination
        batch_returns = torch.zeros(batch_size, num_rounds).to(self.device)
        
        # 1. compute total reward of each episode
        for i_b in range(batch_size):
            r_size = len(self.batch_rewards[i_b])
            batch_returns[i_b][:r_size] = self.batch_r(i_b)          

        # 2. compute time-based baseline values
        batch_baselines = torch.mean(batch_returns, dim=0)
        # largest instead of the mean
        # 3. calculate advantages for each transition
        batch_advs = batch_returns - batch_baselines
        
        # 4. normalize advantages within a batch
        eps = np.finfo(np.float32).eps.item()
        adv_mean = batch_advs.mean()
        adv_std = batch_advs.std()
        batch_advs_norm = (batch_advs - adv_mean) / (adv_std + eps)
        
        # 5. calculate loss for each episode in the batch
        for i_b in range(batch_size):
            for round_count in range(num_rounds):
                # check transtions before early termination
                if round_count < len(self.batch_saved_t_log_probs[i_b]):
                    log_prob = self.batch_saved_t_log_probs[i_b][round_count] + self.batch_saved_w_log_probs[i_b][round_count]
                    entropy = self.batch_saved_t_entropy[i_b][round_count] + self.batch_saved_w_entropy[i_b][round_count]
                    adv_n = batch_advs_norm[i_b][round_count]
                    batch_policy_loss[i_b].append(-log_prob * adv_n - self.entropy_coefficient * entropy)
                    # print(-log_prob * adv_n, - self.entropy_coefficient * entropy, batch_policy_loss[i_b][-1])
                        
            if len(batch_policy_loss[i_b]) > 0:
                batch_total_loss.append(
                    torch.stack(batch_policy_loss[i_b]).sum())

        # reset gradients
        self.optimizer.zero_grad()
        
        # sum up over all batches
        total_loss = torch.stack(batch_total_loss).sum()
        loss_np = total_loss.data.cpu().numpy()
        
        # print(self.batch_saved_t_log_probs, total_loss, batch_policy_loss)
        # perform backprop
        total_loss.backward()
        
        # perform Gradient Clipping-by-norm
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm, norm_type=2)
        
        self.optimizer.step()
        
        # reset rewards and action buffer
        for i_b in range(batch_size):
            del self.batch_rewards[i_b][:]
            del self.batch_saved_t_log_probs[i_b][:]
            del self.batch_saved_t_entropy[i_b][:]
            del self.batch_saved_w_log_probs[i_b][:]
            del self.batch_saved_w_entropy[i_b][:]
        return loss_np
    '''
    Compute total reward of each episode
    '''
    def batch_r(self, i_b):
        R = 0.0
        returns = [] # list to save the true values

        for rw in self.batch_rewards[i_b][::-1]:
            # calculate the discounted value
            R = rw + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns).to(self.device)
        return returns

    '''
    Adjust learning rate using lr_scheduler
    '''
    def adjust_lr(self, metrics=0.0):
        self.lr_scheduler.step()

class GreedyBaselineScheduler():
    """ Policy Gradient Scheduler basex on Repair21

    """
    def __init__(self, device = torch.device("cpu"),
                 nn = 'hybrid', 
                 gamma=0.95, lr=1e-4,
                 weight_decay=1e-5, lmbda = 0.95,
                 milestones=[30, 80], lr_gamma=0.1,
                 entropy_coefficient=0.0,
                 selection_mode='sample',
                 verbose='none'):
        self.device = device
        self.nn = nn
        # self.model = HybridPolicyNet(selection_mode=selection_mode) #.to(self.device)
        # self.model = HybridPolicyNetSparse(selection_mode=selection_mode) #.to(self.device)
        self.model = HybridPolicyNetUpdateSelected(selection_mode=selection_mode, verbose=verbose, device=device).to(self.device)
        # self.model = HybridPolicyNetUpdateSelectedNonBias(selection_mode=selection_mode) #.to(self.device)
        if nn == 'hetgat':
            self.model = HybridPolicyNetNonRecursive(selection_mode=selection_mode, verbose=verbose, device=device).to(self.device)
        
        self.saved_log_probs = []
        self.rewards = []
        self.gamma = gamma
        self.lmbda = lmbda
        self.lr = lr
        self.weight_decay = weight_decay
        self.entropy_coefficient = entropy_coefficient
        
        self.eps = np.finfo(np.float32).eps.item()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones = milestones,
                                        gamma = lr_gamma)
    
    def load_checkpoint(self, trained_checkpoint, retain_old: bool = True):
        cp = torch.load(trained_checkpoint)
        self.model.load_state_dict(cp['policy_net_state_dict'])
        self.optimizer.load_state_dict(cp['optimizer_state_dict'])
        if retain_old:
            self.lr_scheduler.load_state_dict(cp['scheduler_state_dict'])
        else:
            print("Optimizer", self.optimizer.state_dict['lr'], cp['optimizer_state_dict'])
            # Scheduler Import relevant parts of the scheduler:
            relevant_lr_scheduler_state_dict = self.lr_scheduler.state_dict()
            relevant_lr_scheduler_state_dict['last_epoch'] = cp['scheduler_state_dict']['last_epoch']
            relevant_lr_scheduler_state_dict['_step_count'] = cp['scheduler_state_dict']['_step_count']
            self.lr_scheduler.load_state_dict(relevant_lr_scheduler_state_dict)
            print(self.lr_scheduler.state_dict())
        return cp['i_batch'] + 1

    def get_variables(self, env):
        # Unscheduled Tasks
        num_tasks = env.problem.num_tasks
        num_robots = env.team.num_robots
        num_humans = env.team.num_humans
        curr_g = copy.deepcopy(env.halfDG)
        curr_partials = copy.deepcopy(env.partials)
        curr_partialw = copy.deepcopy(env.partialw)
        durs = copy.deepcopy(env.dur)
        # Act Robot is not used for this model of Scheduler
        act_robot = 0
        unsch_tasks = np.array(action_helper_rollout(num_tasks, curr_partialw), dtype=np.int64)
        # Graph Neural Network
        g = build_hetgraph(curr_g,
                            num_tasks, num_robots, num_humans,
                            durs,
                            curr_partials, unsch_tasks)
        # Feature Dictionary
        num_actions = len(unsch_tasks)
        feat_dict = hetgraph_node_helper(curr_g.number_of_nodes(), 
                                         curr_partialw, 
                                         curr_partials,
                                         # transition.locs, 
                                         durs, 
                                         # map_width, 
                                         num_robots + num_humans, num_actions)
        
        feat_dict_tensor = {}
        for key in feat_dict:
            feat_dict_tensor[key] = torch.Tensor(feat_dict[key]).to(device)
        
        return g, feat_dict_tensor, unsch_tasks

    def select_action(self, env):
        """Generate a Schedule as Action for a MultiRoundEnvironment
        Args:
            env (SingleRoundScheduler): Single-Round Scheduler Environment
        """
        # No Grad
        with torch.no_grad():
            schedule = self.model(env)
        return schedule
        
    def initialize_batch(self, batch_size):
        self.batch_saved_t_log_probs = [[] for i in range(batch_size)]
        self.batch_saved_t_entropy = [[] for i in range(batch_size)]
        self.batch_saved_w_log_probs = [[] for i in range(batch_size)]
        self.batch_saved_w_entropy = [[] for i in range(batch_size)]
        self.batch_rewards = [[] for i in range(batch_size)]
    
    def update_model(self, scheduler):
        # Clone Model
        self.model.load_state_dict(copy.deepcopy(scheduler.model.state_dict()))
            
    def batch_select_action(self, env, i_b):
        """Batch Selection of Action
        Args:
            env
        """
        # Reset the model log_prob buffer to save space.
        ## Task Classifier
        self.model.task_classifier.saved_log_probs = []
        self.model.task_classifier.saved_entropy = []
        ## Worker Classifier
        self.model.worker_classifier.saved_log_probs = []
        self.model.worker_classifier.saved_entropy = []
        
        # produced variables required for the model
        # Generate a schedule
        schedule = self.model(env)
        # Add the log probabilities to the batch data
        self.batch_saved_t_log_probs[i_b].append(self.model.task_classifier.saved_log_probs[-1])
        self.batch_saved_w_log_probs[i_b].append(self.model.worker_classifier.saved_log_probs[-1])
        # Add the entropy to the batch data
        self.batch_saved_t_entropy[i_b].append(self.model.task_classifier.saved_entropy[-1])
        self.batch_saved_w_entropy[i_b].append(self.model.worker_classifier.saved_entropy[-1])
        
        return schedule

    def batch_finish_episode(self, batch_size, num_rounds = 1, max_norm=0.75, baseline_rewards=None):        
        '''
        Batch version
        '''
        batch_policy_loss = [[] for i in range(batch_size)]
        batch_total_loss = []
        
        # zero-pad episodes with early termination
        batch_returns = torch.zeros(batch_size, num_rounds).to(self.device)
        
        # 1. compute total reward of each episode
        for i_b in range(batch_size):
            r_size = len(self.batch_rewards[i_b])
            batch_returns[i_b][:r_size] = self.batch_r(i_b)          

        # 2. compute time-based baseline values
        batch_baselines = torch.mean(batch_returns, dim=0)
        if baseline_rewards is not None:
            batch_baselines = baseline_rewards
            
        # largest instead of the mean
        # 3. calculate advantages for each transition
        batch_advs = batch_returns - batch_baselines
        
        # 4. normalize advantages within a batch
        eps = np.finfo(np.float32).eps.item()
        adv_mean = batch_advs.mean()
        adv_std = batch_advs.std()
        batch_advs_norm = (batch_advs - adv_mean) / (adv_std + eps)
        
        # 5. calculate loss for each episode in the batch
        for i_b in range(batch_size):
            for round_count in range(num_rounds):
                # check transtions before early termination
                if round_count < len(self.batch_saved_t_log_probs[i_b]):
                    log_prob = self.batch_saved_t_log_probs[i_b][round_count] + self.batch_saved_w_log_probs[i_b][round_count]
                    entropy = self.batch_saved_t_entropy[i_b][round_count] + self.batch_saved_w_entropy[i_b][round_count]
                    adv_n = batch_advs_norm[i_b][round_count]
                    batch_policy_loss[i_b].append(-log_prob * adv_n - self.entropy_coefficient * entropy)
                        
            if len(batch_policy_loss[i_b]) > 0:
                batch_total_loss.append(
                    torch.stack(batch_policy_loss[i_b]).sum())

        # reset gradients
        self.optimizer.zero_grad()
        
        # sum up over all batches
        total_loss = torch.stack(batch_total_loss).sum()
        loss_np = total_loss.data.cpu().numpy()
        
        # print(self.batch_saved_t_log_probs, total_loss, batch_policy_loss)
        # perform backprop
        total_loss.backward()
        
        # perform Gradient Clipping-by-norm
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm, norm_type=2)
        
        self.optimizer.step()
        
        # reset rewards and action buffer
        for i_b in range(batch_size):
            del self.batch_rewards[i_b][:]
            del self.batch_saved_t_log_probs[i_b][:]
            del self.batch_saved_t_entropy[i_b][:]
            del self.batch_saved_w_log_probs[i_b][:]
            del self.batch_saved_w_entropy[i_b][:]

        return loss_np
    
    def get_baseline_rewards(self, num_rounds):        
        # zero-pad episodes with early termination
        batch_returns = torch.zeros(num_rounds).to(self.device)
        # 1. compute total reward of each episode
        r_size = len(self.batch_rewards[0])
        batch_returns[:r_size] = self.batch_r(0)
        del self.batch_rewards[0][:]
        return batch_returns

    '''
    Compute total reward of each episode
    '''
    def batch_r(self, i_b):
        R = 0.0
        returns = [] # list to save the true values

        for rw in self.batch_rewards[i_b][::-1]:
            # calculate the discounted value
            R = rw + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns).to(self.device)
        return returns

    '''
    Adjust learning rate using lr_scheduler
    '''
    def adjust_lr(self, metrics=0.0):
        self.lr_scheduler.step()

class SoftScheduler():
    """ Policy Gradient Scheduler basex on Repair21

    """
    def __init__(self, scheduler):
        self.scheduler = scheduler        
        pass

    def select_action(self, env):
        """Generate a Schedule as Action for a MultiRoundEnvironment
        Args:
            env (SingleRoundScheduler): Single-Round Scheduler Environment
        """
        schedule = self.scheduler.select_action(env)
        
    
if __name__ == '__main__':
    from lr_hybrid_scheduler_train import fill_demo_data
    from utils_hybrid import ReplayMemory, Transition, action_helper_rollout
    from utils_hybrid import hetgraph_node_helper, build_hetgraph

    folder = 'tmp/test1'
    memory, envs = fill_demo_data(folder, 0, 2, 0.99)
    env = envs[1]

    # print(env.halfDG.edges)
    # print(memory.memory[0].curr_g.edges())
    scheduler = PGScheduler(device=torch.device('cpu',0))
    for i in range(2):
        multi_round_env = MultiRoundSchedulingEnv(env.problem, env.team)
        schedule = scheduler.select_action(env)
        success, reward, done, _ = multi_round_env.step(schedule)
        scheduler.rewards.append(reward)
        
        print(success, reward, done)

    # Calculate Policy Loss
    scheduler.finish_episode()
    
