# -*- coding: utf-8 -*-

import argparse
import torch
import os
from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call
from datetime import datetime
from benchmark_utils import *

current_time = str(datetime.now().strftime("%Y%m%d-%H%M%S"))

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoints', default=None)
parser.add_argument('--cpu', default=False, action='store_true')
parser.add_argument('--val-start-no', default=1, type=int)
parser.add_argument('--val-end-no', default=800, type=int)
parser.add_argument('--test-start-no', default=801, type=int)
parser.add_argument('--test-end-no', default=1000, type=int)
parser.add_argument('--thread-pool-size', default=10, type=int)
parser.add_argument('--versions', default='v0,v1,v2,v3')
parser.add_argument('--skip-test', default=False, action='store_true')
parser.add_argument('--skip-perf-eval', default=False, action='store_true')
parser.add_argument('--data-folders', default='r2t20_002,r2t50_001,r5t20_001,r10t20_001')
args = parser.parse_args()

device = torch.device('cpu' if args.cpu else 'cuda')

checkpoints = []
data_folders = args.data_folders.split(',')
data_folder_prefix = '../gen/'

for file in os.listdir(args.checkpoints):
    if file.endswith('.tar'):
        checkpoints.append(os.path.join(args.checkpoints, file))

print('Checkpoints:\n' + '\n'.join(checkpoints))

tasks = []
test_tasks = []

for checkpoint in checkpoints:
    for data_folder in data_folders:
        for version in args.versions.split(','):
            tasks.append((checkpoint, data_folder, version))


def get_benchmark_commands(start_no, end_no, tasks):
    return [f'python het_edf.py'
            f'{" --cpu" if args.cpu else ""}'
            f' --checkpoint {checkpoint}'
            f' --data-path {data_folder_prefix}{data_folder}'
            f' --version {version}'
            f' --results-folder-suffix {current_time}'
            f' --start-no {start_no}'
            f' --end-no {end_no}'
            for checkpoint, data_folder, version in tasks]


def get_perf_eval_commands(start_no, end_no):
    return [f'python perf_eval.py'
            f' --data {data_folder_prefix}{data_folder}'
            f' --v0-results-folder {get_results_folder_name("v0", data_folder, start_no, end_no, current_time)}'
            f' --start-no {start_no}'
            f' --end-no {end_no}'
            for data_folder in data_folders]


val_commands = get_benchmark_commands(args.val_start_no, args.val_end_no, tasks)
test_commands = get_benchmark_commands(args.test_start_no, args.test_end_no, tasks)
val_perf_eval_commands = get_perf_eval_commands(args.val_start_no, args.val_end_no)
test_perf_eval_commands = get_perf_eval_commands(args.test_start_no, args.test_end_no)

pool = Pool(args.thread_pool_size)


def run_tasks(commands, tasks_name):
    print('\n'.join(commands))

    commands_done = 0
    for i, returncode in enumerate(pool.imap(partial(call, shell=True), commands)):
        if returncode != 0:
            print(f'Command {commands[i]} failed: {returncode}')

        commands_done += 1
        print(f'\n{commands_done}/{len(commands)} {tasks_name} tasks done\n')


run_tasks(val_commands if args.skip_test else val_commands + test_commands, 'benchmark')
if not args.skip_perf_eval:
    run_tasks(val_perf_eval_commands + test_perf_eval_commands, 'perf eval')
