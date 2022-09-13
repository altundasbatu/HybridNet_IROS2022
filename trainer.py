# -*- coding: utf-8 -*-
"""
Created on Fri October 14 12:47:42 2021

@author: baltundas

Batched PG Trainer
"""

import os
import time
import random
import numpy as np
import pickle
import argparse

import torch
import torch.nn as nn
from torch.distributions import Categorical

import torch.optim as optim

from env.mrc_problem import MRCProblem
from env.hybrid_team import HybridTeam
from scheduler import PGScheduler, GreedyBaselineScheduler

from env.scheduling_env import SchedulingEnv
from env.multi_round_scheduling_env import MultiRoundSchedulingEnv

import matplotlib.pyplot as plt

class PGTrainer(object):
    def __init__(self, scheduler: PGScheduler, 
                        folder = 'tmp/pg_test', 
                        resume_training=False, 
                        training_checkpoint=2000,
                        checkpoint_location=None,
                        detach_gap: int = 10):

        self.detach_gap = detach_gap

        if not os.path.exists(folder):
            os.makedirs(folder)
        self.folder = folder
        if checkpoint_location is None:
            self.checkpoint_location = folder
        else:
            self.checkpoint_location = checkpoint_location

        self.scheduler = scheduler
        self.device = self.scheduler.device
    
        os.makedirs(self.folder, exist_ok=True)
        os.makedirs(self.checkpoint_location, exist_ok=True)
        
        self.efficiency_metrics_folder = self.checkpoint_location + '/efficiency_metrics.txt'
        self.feasible_solution_folder = self.checkpoint_location + '/feasible_solution_count.txt'
        
        self.efficiency_metrics = []
        self.feasible_solution_count = []

        if resume_training:
            trained_checkpoint = self.checkpoint_location + "/checkpoint_{:05d}.tar".format(training_checkpoint)
            self.start_episode = self.scheduler.load_checkpoint(trained_checkpoint, retain_old = False)
            try:
                self.efficiency_metrics = np.loadtxt(self.efficiency_metrics_folder)[:self.start_episode - 1].tolist()
                # print(self.efficiency_metrics.shape, self.start_episode - 1)
                self.feasible_solution_count = np.loadtxt(self.feasible_solution_folder)[:self.start_episode - 1].tolist()
                # print(self.efficiency_metrics, len(self.efficiency_metrics), len(self.feasible_solution_count))
                # Overwrite the record to reflect only up to the checkpoint
                np.savetxt(self.efficiency_metrics_folder, np.array(self.efficiency_metrics))
                self.efficiency_metrics = []
                np.savetxt(self.feasible_solution_folder, np.array(self.feasible_solution_count))
                self.feasible_solution_count = []
            except Exception as e:
                print(e)
                pass
        else:
            self.start_episode = 1
            # Save Initial Checkpoint 0
            checkpoint_path = self.checkpoint_location+'/checkpoint_{:05d}.tar'.format(0)
            torch.save({
                'i_batch': 0,
                'policy_net_state_dict': self.scheduler.model.state_dict(),
                'optimizer_state_dict': self.scheduler.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.lr_scheduler.state_dict(),
                'loss': []
            }, checkpoint_path)
            print('checkpoint saved to ' + checkpoint_path)
        # open files as append
        self.efficiency_metrics_f = open(self.efficiency_metrics_folder, 'a')
        self.feasible_solution_f = open(self.feasible_solution_folder, 'a')

        self.scheduler.initialize_batch(BATCH_SIZE)


    def train(self, num_humans, num_robots, num_tasks, total_episodes, num_rounds, max_norm=0.75, human_learning=True, from_file=False, infeasible_coefficient=2.0, start_problem = 1, end_problem = 1000, noise = False, estimator=False, est_noise=False):
        loss_history = []
        efficiency_record = self.efficiency_metrics
        feasible_solution_count = self.feasible_solution_count
        infeasible_generation_count = 0
        for i_batch in range(self.start_episode, total_episodes + 1):
            '''
            Initialize
                Episodes within a batch use the same length/simulation_time
            '''
            # print("Last LR", self.scheduler.lr_scheduler.state_dict()['_last_lr'])
            start_t = time.time()
            batch_reward = []
            
            print('Training batch: {:d}'.format(i_batch))
            # print('Learning Rate: ', self.scheduler.lr_scheduler.state_dict())
            '''
            Use the same initialized env for the batch. Since actions are sampled, this allows for random batching for the same environment, producing different actions.
            '''

            problem_file_name = self.folder + "/problem_" +  format(i_batch, '04')
            problem = None
            if from_file:
                # randomly select from the files
                max_stored_episode = end_problem - start_problem + 1
                id_ = (i_batch - 1) % max_stored_episode + start_problem
                problem_file_name = self.folder + "/problem_" +  format(id_, '04')
                problem = MRCProblem(fname = problem_file_name, max_deadline_multiplier=infeasible_coefficient, noise=noise)
            else:
                # Generate a Feasible Problem
                num_tasks_chosen = num_tasks
                if isinstance(num_tasks, list):
                    num_tasks_chosen = random.randint(num_tasks[0],num_tasks[1])
                num_humans_chosen = num_humans
                if isinstance(num_humans, list):
                    num_humans_chosen = random.randint(num_humans[0],num_humans[1])
                num_robots_chosen = num_robots
                if isinstance(num_robots, list):
                    num_robots_chosen = random.randint(num_robots[0],num_robots[1])
                problem = MRCProblem(num_tasks_chosen, num_robots_chosen, num_humans_chosen, max_deadline_multiplier=infeasible_coefficient, noise=noise)
                generated_feasible_problem = problem.generate_solution(problem_file_name)
                while not generated_feasible_problem:
                    infeasible_generation_count += 1
                    print("Generated Infeasible Problem, generating again...", i_batch)
                    problem = MRCProblem(num_tasks_chosen, num_robots_chosen, num_humans_chosen, max_deadline_multiplier=infeasible_coefficient, noise=noise)
                    generated_feasible_problem = problem.generate_solution(problem_file_name)
                problem.save_to_file(problem_file_name)
            # Create a Team
            team = HybridTeam(problem)

            # buffer_name = self.folder+'/buffer_env.pkl'
            # with open(buffer_name, 'wb') as f:
            #     pickle.dump(r, f)
            
            # Create multiple instances of the same environment, since the environments update after every round
            multi_round_envs = [MultiRoundSchedulingEnv(problem, team) for i in range(BATCH_SIZE)]
            
            feasible_solution_count.append(0)
            '''
            Run multiple episodes on the same environment, with each step changing the human model based on the repetitions of tasks
            '''
            for i_b in range(BATCH_SIZE):
                for step_count in range(num_rounds):
                    env = None
                    if estimator:
                        env = multi_round_envs[i_b].get_estimate_environment(est_noise=est_noise)
                    else:
                        env = multi_round_envs[i_b].get_actual_environment(human_noise=noise)
                    schedule = self.scheduler.batch_select_action(env, i_b)
                    success, reward, done, _ = multi_round_envs[i_b].step(schedule, human_learning=human_learning, evaluate=estimator, human_noise=noise, estimator_noise=est_noise)
                    if success: # the generated schedule is feasible
                        feasible_solution_count[-1] += 1
                    self.scheduler.batch_rewards[i_b].append(reward)
                    print('reward: {:.4f}'.format(reward), end='\r')
            # print(self.scheduler.batch_rewards)
            average_makespan = -np.sum(self.scheduler.batch_rewards)/(BATCH_SIZE * num_rounds)
            efficiency_metric = 1.0 - average_makespan/problem.max_deadline
            efficiency_record.append(efficiency_metric)
            
                
            loss = self.scheduler.batch_finish_episode(BATCH_SIZE, num_rounds, max_norm=max_norm)
            loss_history.append(loss)
            
            '''
            Perform training when all batch episodes finish
            '''
            if i_batch > 1:
                self.scheduler.adjust_lr()

            end_t = time.time()
            print('[Batch {}], loss: {:e}, time: {:.3f} s'.
                format(i_batch, loss_history[-1], end_t - start_t))

            '''
            Save checkpoints
            '''
            if i_batch % 10 == 0:
                checkpoint_path = self.checkpoint_location+'/checkpoint_{:05d}.tar'.format(i_batch)
                torch.save({
                    'i_batch': i_batch,
                    'policy_net_state_dict': self.scheduler.model.state_dict(),
                    'optimizer_state_dict': self.scheduler.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.lr_scheduler.state_dict(),
                    'loss': loss_history
                }, checkpoint_path)
                print('checkpoint saved to '+checkpoint_path)
                
                # # Update the Plot
                # epochs = range(len(efficiency_record))
                # plt.plot(epochs, efficiency_record)
                # efficiency_graph_path = self.folder + '/efficiency_graph.png'
                # plt.savefig(efficiency_graph_path)
                # plt.clf()
                # plt.plot(epochs, feasible_solution_count)
                # efficiency_graph_path = self.folder + '/feasible_count_graph.png'
                # plt.savefig(efficiency_graph_path)
                # plt.clf()
                
                # Save the Latest Efficiency Raw Data for future graphing
                self.efficiency_metrics_f = open(self.efficiency_metrics_folder, 'a')
                np.savetxt(self.efficiency_metrics_f, np.array(efficiency_record))
                efficiency_record = [] # reset buffer
                self.efficiency_metrics_f.close()
                # Save the Latest Feasible Solution Counts for future graphing
                self.feasible_solution_f = open(self.feasible_solution_folder, 'a')
                np.savetxt(self.feasible_solution_f, np.array(feasible_solution_count))
                feasible_solution_count = []
                self.feasible_solution_f.close()

        
        # # Save the Latest Efficiency Raw Data for future graphing
        # np.savetxt(self.efficiency_metrics_folder, np.array(efficiency_record))
        # np.savetxt(self.feasible_solution_folder, np.array(feasible_solution_count))
        # print("Infeasible generation count:", infeasible_generation_count, "for", total_episodes, "episodes")
        print('Complete')

class GreedyBaselineTrainer(object):
    """Greedy Rollout Baseline Trainer
    Based on: Kool, W.; van Hoof, H.; and Welling, M. 2019.  Attention,Learn to Solve Routing Problems! arXiv:1803.08475
    """
    def __init__(self, scheduler: GreedyBaselineScheduler,
                        folder = 'tmp/pg_test', 
                        resume_training=False, 
                        training_checkpoint=500,
                        checkpoint_location=None,
                        detach_gap: int = 10,
                        baseline_update_rate: int = 5):

        self.detach_gap = detach_gap

        if not os.path.exists(folder):
            os.makedirs(folder)
        self.folder = folder
        if checkpoint_location is None:
            self.checkpoint_location = folder
        else:
            self.checkpoint_location = checkpoint_location

        self.scheduler = scheduler
        self.baseline_scheduler = GreedyBaselineScheduler(self.scheduler.device, nn=self.scheduler.nn, selection_mode='argmax')
        self.baseline_update_rate = baseline_update_rate
        
        self.device = self.scheduler.device
        
        os.makedirs(self.folder, exist_ok=True)
        os.makedirs(self.checkpoint_location, exist_ok=True)
        
        self.efficiency_metrics_folder = self.checkpoint_location + '/efficiency_metrics.txt'
        self.feasible_solution_folder = self.checkpoint_location + '/feasible_solution_count.txt'
        
        self.efficiency_metrics = []
        self.feasible_solution_count = []

        if resume_training:
            trained_checkpoint = self.checkpoint_location + "/checkpoint_{:05d}.tar".format(training_checkpoint)
            self.start_episode = self.scheduler.load_checkpoint(trained_checkpoint, retain_old = True)
            try:
                self.efficiency_metrics = np.loadtxt(self.efficiency_metrics_folder)[:self.start_episode - 1].tolist()
                # print(self.efficiency_metrics.shape, self.start_episode - 1)
                self.feasible_solution_count = np.loadtxt(self.feasible_solution_folder)[:self.start_episode - 1].tolist()
                # print(self.efficiency_metrics, len(self.efficiency_metrics), len(self.feasible_solution_count))
                # Overwrite the record to reflect only up to the checkpoint
                np.savetxt(self.efficiency_metrics_folder, np.array(self.efficiency_metrics))
                self.efficiency_metrics = []
                np.savetxt(self.feasible_solution_folder, np.array(self.feasible_solution_count))
                self.feasible_solution_count = []
            except Exception as e:
                print(e)
                pass
        else:
            self.start_episode = 1
            # Save Initial Checkpoint 0
            checkpoint_path = self.checkpoint_location+'/checkpoint_{:05d}.tar'.format(0)
            torch.save({
                'i_batch': 0,
                'policy_net_state_dict': self.scheduler.model.state_dict(),
                'optimizer_state_dict': self.scheduler.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.lr_scheduler.state_dict(),
                'loss': []
            }, checkpoint_path)
            print('checkpoint saved to ' + checkpoint_path)
        # Rebase the baseline scheduler to be same as the main scheduler for stability (for both continue training and starting from scratch)
        self.baseline_scheduler.update_model(self.scheduler)
        
        # open files as append
        self.efficiency_metrics_f = open(self.efficiency_metrics_folder, 'a')
        self.feasible_solution_f = open(self.feasible_solution_folder, 'a')
        # Initialize Batch
        self.scheduler.initialize_batch(BATCH_SIZE)
        self.baseline_scheduler.initialize_batch(1)


    def train(self, num_humans, num_robots, num_tasks, total_episodes, num_rounds, max_norm=0.75, human_learning=True, from_file=False, infeasible_coefficient=2.0, start_problem = 1, end_problem = 1000, noise = False, estimator=False, est_noise=False):
        loss_history = []
        efficiency_record = self.efficiency_metrics
        feasible_solution_count = self.feasible_solution_count
        infeasible_generation_count = 0
        for i_batch in range(self.start_episode, total_episodes + 1):
            '''
            Initialize
                Episodes within a batch use the same length/simulation_time
            '''
            start_t = time.time()
            batch_reward = []
            
            print('Training batch: {:d}'.format(i_batch))
            '''
            Use the same initialized env for the batch. Since actions are sampled, this allows for random batching for the same environment, producing different actions.
            '''

            problem_file_name = self.folder + "/problem_" +  format(i_batch, '04')
            problem = None
            if from_file:
                # randomly select from the files
                max_stored_episode = end_problem - start_problem + 1
                id_ = (i_batch - 1) % max_stored_episode + start_problem
                problem_file_name = self.folder + "/problem_" +  format(id_, '04')
                problem = MRCProblem(fname = problem_file_name, max_deadline_multiplier=infeasible_coefficient, noise=noise)
            else:
                # Generate a Feasible Problem
                num_tasks_chosen = num_tasks
                if isinstance(num_tasks, list):
                    num_tasks_chosen = random.randint(num_tasks[0],num_tasks[1])
                num_humans_chosen = num_humans
                if isinstance(num_humans, list):
                    num_humans_chosen = random.randint(num_humans[0],num_humans[1])
                num_robots_chosen = num_robots
                if isinstance(num_robots, list):
                    num_robots_chosen = random.randint(num_robots[0],num_robots[1])
                problem = MRCProblem(num_tasks_chosen, num_robots_chosen, num_humans_chosen, max_deadline_multiplier=infeasible_coefficient, noise=noise)
                generated_feasible_problem = problem.generate_solution(problem_file_name)
                while not generated_feasible_problem:
                    infeasible_generation_count += 1
                    print("Generated Infeasible Problem, generating again...", i_batch)
                    problem = MRCProblem(num_tasks_chosen, num_robots_chosen, num_humans_chosen, max_deadline_multiplier=infeasible_coefficient, noise=noise)
                    generated_feasible_problem = problem.generate_solution(problem_file_name)
                problem.save_to_file(problem_file_name)
            # Create a Team
            team = HybridTeam(problem)

            # Policy Gradient
            # Create multiple instances of the same environment, since the environments update after every round
            multi_round_envs = [MultiRoundSchedulingEnv(problem, team) for i in range(BATCH_SIZE)]
            baseline_env = MultiRoundSchedulingEnv(problem, team)
            
            '''
            Run multiple episodes on the same environment, with each step changing the human model based on the repetitions of tasks
            '''
            for i_b in range(BATCH_SIZE):
                for step_count in range(num_rounds):
                    env = None
                    if estimator:
                        env = multi_round_envs[i_b].get_estimate_environment(est_noise=est_noise)
                    else:
                        env = multi_round_envs[i_b].get_actual_environment(human_noise=noise)
                    schedule = self.scheduler.batch_select_action(env, i_b)
                    success, reward, done, _ = multi_round_envs[i_b].step(schedule, human_learning=human_learning, evaluate=estimator, human_noise=noise, estimator_noise=est_noise)
                    # if success: # the generated schedule is feasible
                    self.scheduler.batch_rewards[i_b].append(reward)
                    print('reward: {:.4f}'.format(reward), end='\r')
            
            feasible_solution_count.append(0)
            # Greedy Baseline            
            for step_count in range(num_rounds):
                env = None
                if estimator:
                    env = baseline_env.get_estimate_environment(est_noise=est_noise)
                else:
                    env = baseline_env.get_actual_environment(human_noise=noise)
                schedule = self.baseline_scheduler.batch_select_action(env, 0)
                success, reward, done, _ = baseline_env.step(schedule, human_learning=human_learning, evaluate=estimator, human_noise=noise, estimator_noise=est_noise)
                if success:
                    feasible_solution_count[-1] += 1
                    print('Baseline Feasible', end='\r')
                self.baseline_scheduler.batch_rewards[0].append(reward)
                print('Baseline Reward: {:.4f}'.format(reward), end='\r')
            average_baseline_makespan = -np.sum(self.baseline_scheduler.batch_rewards)/(num_rounds)
            efficiency_metric = 1.0 - average_baseline_makespan/problem.max_deadline
            efficiency_record.append(efficiency_metric)
            baseline_rewards = self.baseline_scheduler.get_baseline_rewards(num_rounds)
            loss = self.scheduler.batch_finish_episode(BATCH_SIZE, num_rounds, max_norm=max_norm, baseline_rewards=baseline_rewards)
            loss_history.append(loss)
            
            if i_batch % self.baseline_update_rate == 0:
                # copy 
                self.baseline_scheduler.update_model(self.scheduler)
            '''
            Perform training when all batch episodes finish
            '''
            if i_batch > 1:
                self.scheduler.adjust_lr()

            end_t = time.time()
            print('[Batch {}], loss: {:e}, time: {:.3f} s'.
                format(i_batch, loss_history[-1], end_t - start_t))

            '''
            Save checkpoints
            '''
            if i_batch % 10 == 0:
                checkpoint_path = self.checkpoint_location+'/checkpoint_{:05d}.tar'.format(i_batch)
                torch.save({
                    'i_batch': i_batch,
                    'policy_net_state_dict': self.scheduler.model.state_dict(),
                    'optimizer_state_dict': self.scheduler.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.lr_scheduler.state_dict(),
                    'loss': loss_history
                }, checkpoint_path)
                print('checkpoint saved to '+checkpoint_path)
                
                # # Update the Plot
                # epochs = range(len(efficiency_record))
                # plt.plot(epochs, efficiency_record)
                # efficiency_graph_path = self.folder + '/efficiency_graph.png'
                # plt.savefig(efficiency_graph_path)
                # plt.clf()
                # plt.plot(epochs, feasible_solution_count)
                # efficiency_graph_path = self.folder + '/feasible_count_graph.png'
                # plt.savefig(efficiency_graph_path)
                # plt.clf()
                
                # Save the Latest Efficiency Raw Data for future graphing
                self.efficiency_metrics_f = open(self.efficiency_metrics_folder, 'a')
                np.savetxt(self.efficiency_metrics_f, np.array(efficiency_record))
                efficiency_record = [] # reset buffer
                self.efficiency_metrics_f.close()
                # Save the Latest Feasible Solution Counts for future graphing
                self.feasible_solution_f = open(self.feasible_solution_folder, 'a')
                np.savetxt(self.feasible_solution_f, np.array(feasible_solution_count))
                feasible_solution_count = []
                self.feasible_solution_f.close()
        
        # # Save the Latest Efficiency Raw Data for future graphing
        # np.savetxt(self.efficiency_metrics_folder, np.array(efficiency_record))
        # np.savetxt(self.feasible_solution_folder, np.array(feasible_solution_count))
        # print("Infeasible generation count:", infeasible_generation_count, "for", total_episodes, "episodes")
        print('Complete')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--mode', type=str, default="pg")
    parser.add_argument('--nn', type=str, default="hybridnet")
    parser.add_argument('--folder', type=str, default="tmp/small_training_set")
    parser.add_argument('--start-problem', type=int, default=1)
    parser.add_argument('--end-problem', type=int, default=2000)
    parser.add_argument('--checkpoint', type=str, default="tmp/small_training_set/checkpoints_21_pg")
    parser.add_argument('--resume-cp', type=int, default=-1)
    parser.add_argument('--epoch', type=int, default=1000)
    
    parser.add_argument('--human-noise', action='store_true')
    parser.set_defaults(human_noise=False)
    parser.add_argument('--estimator', action='store_true')
    parser.set_defaults(estimator=False)
    parser.add_argument('--estimator_noise', action='store_true')
    parser.set_defaults(estimator_noise=False)

    parser.add_argument('--num-rounds', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=2e-3) # 8e-3
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--milestones', nargs='+', type=int, default=[4000, 8000, 12000, 16000, 20000])
    parser.add_argument('--lr-gamma', type=float, default=0.5)
    parser.add_argument('--entropy-coefficient', type=float, default=0.1)
    parser.add_argument('--infeasible-coefficient', type=float, default=2.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--verbose', default='none', type=str)
    args = parser.parse_args()
    
    # random seed
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    # environment
    mode = args.mode
    nn = args.nn
    verbose = args.verbose
    
    problem_folder = args.folder
    start_problem = args.start_problem
    end_problem = args.end_problem

    # training parameters
    loc = args.checkpoint
    
    resume_training = False
    resume_cp = args.resume_cp
    if resume_cp >= 0: # if an integer is given, continue from there
        resume_training = True

    total_episodes = args.epoch
    
    real_noise = args.human_noise
    estimator = args.estimator
    est_noise = args.estimator_noise
    
    NUM_ROUNDS = args.num_rounds
    BATCH_SIZE = args.batch_size

    GAMMA = args.gamma
    lr = args.lr
    weight_decay = args.weight_decay
    milestones = args.milestones
    lr_gamma = args.lr_gamma
    entropy_coefficient = args.entropy_coefficient
    infeasible_coefficient = args.infeasible_coefficient
    device = args.device
    device_id = args.device_id
    
    if mode == 'pg':
        # Train Policy Gradient
        scheduler = PGScheduler(device=torch.device(device, device_id), 
                                nn=nn,
                                gamma=GAMMA, lr=lr,
                                weight_decay=weight_decay, lmbda = 0.95,
                                milestones=milestones, lr_gamma=lr_gamma,
                                entropy_coefficient=entropy_coefficient,
                                selection_mode='sample', verbose=verbose)
        trainer = PGTrainer(scheduler,
                            folder = problem_folder, 
                            resume_training=resume_training, 
                            training_checkpoint=resume_cp,
                            checkpoint_location=loc)
        
        trainer.train(2, 2, [9, 11], total_episodes, NUM_ROUNDS, from_file=True, start_problem=start_problem, end_problem=end_problem, noise = real_noise, estimator=estimator, est_noise=est_noise)
    elif mode == 'gb':
        # Train Greedy Baseline
        scheduler = GreedyBaselineScheduler(device=torch.device(device, device_id), 
                                            nn=nn,
                                            gamma=GAMMA, lr=lr,
                                            weight_decay=weight_decay, lmbda = 0.95,
                                            milestones=milestones, lr_gamma=lr_gamma,
                                            entropy_coefficient=entropy_coefficient, verbose=verbose)
        trainer = GreedyBaselineTrainer(scheduler,
                                        folder = problem_folder, 
                                        resume_training=resume_training, 
                                        training_checkpoint=resume_cp,
                                        checkpoint_location=loc)
        
        trainer.train(2, 2, [9, 11], total_episodes, NUM_ROUNDS, from_file=True, start_problem=start_problem, end_problem=end_problem, noise = real_noise, estimator=estimator, est_noise=est_noise)    
    else:
        pass
    