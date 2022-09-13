import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

import torch

from scheduler import PGScheduler
from env.mrc_problem import MRCProblem
from env.hybrid_team import HybridTeam

from env.scheduling_env import SchedulingEnv
from env.multi_round_scheduling_env import MultiRoundSchedulingEnv

if __name__ == '__main__':
	"""
	"""

	parser = argparse.ArgumentParser()
	parser.add_argument('--cp', type=str, default="tmp/small_training_set/checkpoints_13_pg")
	parser.add_argument('--cp-version', type=int, default=3200)
	parser.add_argument('--problem-dir', type=str, default="tmp/small_test_set")
	parser.add_argument('--problem-num', type=int, default=1)
	parser.add_argument('--mode', type=str, default="argmax")

    # Batch Count
	parser.add_argument('--batch-size', type=int, default=8)
    # Round Count
	parser.add_argument('--num_rounds', type=int, default=4)

	args = parser.parse_args()

	# Checkpoint
	cp_parent = args.cp
	checkpoint = args.cp_version
	checkpoint_folder = cp_parent + "/checkpoint_%05d.tar" % checkpoint

	# Problem
	fname = args.problem_dir + '/problem_' + format(args.problem_num, '04')

	# Selection Mode
	mode = args.mode
	batch_size = args.batch_size
	num_rounds = args.num_rounds
	human_learning = True

	'''
	Load model
	'''
	# scheduler = PGScheduler(device=torch.device('cpu',0))
	scheduler = PGScheduler(device=torch.device('cuda',0), selection_mode=mode)
	scheduler.load_checkpoint(checkpoint_folder)
	print('Loaded: '+checkpoint_folder)


    # load env from data folder
	problem = MRCProblem(fname = fname)
	print(problem.dur)
    # Create a Team
	team = HybridTeam(problem)
                
    # scheduler already being loaded outside of this function    
	if mode == 'sample':
		multi_round_envs = [MultiRoundSchedulingEnv(problem, team) for i in range(batch_size)]
		for i_b in range(batch_size):
			print('Batch {}/{}.'.format(i_b+1, batch_size))
			for step_count in range(num_rounds):
				schedule = scheduler.select_action(multi_round_envs[i_b].get_single_round())
				success, reward, done, makespan = multi_round_envs[i_b].step(schedule, human_learning=human_learning)
			print("Schedule:", schedule)
			print("Durations", problem.dur)
			print("Wait Conditions", problem.wait)
			print("Makespan:", makespan)
	elif mode == 'argmax':
		multi_round_env = MultiRoundSchedulingEnv(problem, team)
		for step_count in range(num_rounds):
			schedule = scheduler.select_action(multi_round_env.get_single_round())
			success, reward, done, makespan = multi_round_env.step(schedule, human_learning=human_learning)
			print("Schedule:", schedule)
			print("Durations", problem.dur)
			print("Wait Conditions", problem.wait)
			print("Makespan:", makespan)

