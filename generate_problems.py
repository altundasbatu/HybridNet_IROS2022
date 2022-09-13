"""
Created on Tuesday November 23 12:47:42 2021

@author: baltundas
"""

import os
import random
import argparse

from env.mrc_problem import MRCProblem
from env.hybrid_team import HybridTeam

def generate_task_allocation_problems(folder, num_humans, num_robots, num_tasks, start, finish):
    os.makedirs(folder, exist_ok=True)
    for i in range(start+1, finish+1):
        num_tasks_chosen = num_tasks
        if isinstance(num_tasks, list):
            num_tasks_chosen = random.randint(num_tasks[0],num_tasks[1])
        num_humans_chosen = num_humans
        if isinstance(num_humans, list):
            num_humans_chosen = random.randint(num_humans[0],num_humans[1])
        num_robots_chosen = num_robots
        if isinstance(num_robots, list):
            num_robots_chosen = random.randint(num_robots[0],num_robots[1])

        file_name = folder + "/problem_" + format(i, '04')
        problem = MRCProblem(num_tasks_chosen, num_robots_chosen, num_humans_chosen)
        success = problem.save_to_file(file_name)
        while not success:
            problem = MRCProblem(num_tasks_chosen, num_robots_chosen, num_humans_chosen)
            success = problem.save_to_file(file_name)
            
def read_test_data(folder, start, finish):
    problems = []
    for i in range(start+1, finish+1):
        file_name = folder + "/problem_" + format(i, '04')
        problem = MRCProblem(fname=file_name)
        problems.append(problem)
    return problems

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str)
    parser.add_argument('--human', int, defualt=2)
    parser.add_argument('--robot', int, default=2)
    parser.add_argument('--task-min', int, default=9)
    parser.add_argument('--task-max', int, default=11)
    parser.add_argument('--start', int, default=0)
    parser.add_argument('--end', int, default=2000)
    
    args = parser.parse_args()
    generate_task_allocation_problems(args.folder, args.human, args.robot, [args.task_min, args.task_max], args.start, args.finish)
    print('Done')