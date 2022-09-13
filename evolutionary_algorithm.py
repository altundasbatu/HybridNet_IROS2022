import numpy as np
import random
import copy
from env.hybrid_team import HybridTeam

def get_swappable_steps(schedule, team: HybridTeam, delta_coefficient = 0.5):
    """Generates a Bi-Directional Map of Swappable Steps.

    Args:
        schedule (list of tuple): Schedule List of (task_id, worker_id, diff)
        team (HybridTeam): HybridTeam containing information about task durations for workers
        delta_coefficient (float, optional): Delat multiplier for Average Task Duration. Defaults to 0.5.

    Returns:
        swappable_steps (Dictionary step_id: list[step_id]): 
    """
    # Generate Task Start Times
    swappable_steps = {s_id: [] for s_id in range(len(schedule))} # a bi-directional map of swappable task list
    worker_current_time = {w_id: 0 for w_id in range(len(team.workers))}
    task_start_times = []
    total_duration = 0
    for t_id, w_id, _ in schedule:
        duration = team.get_duration(t_id - 1, w_id)
        task_start_times.append(worker_current_time[w_id])        
        worker_current_time[w_id] += duration
        total_duration += duration
    average_duration = total_duration / len(schedule)
    delta = delta_coefficient * average_duration # for "some 𝛿 much less than the average task duration"
    arg_order = np.argsort(np.array(task_start_times)) # index matches the schedule order for swapping
    for i in range(len(arg_order)):
        s_id = arg_order[i]
        start_time = task_start_times[s_id]
        # t_id, w_id, _ = schedule[schedule_index]
        for j in range(i + 1, len(arg_order)):
            s_id2 = arg_order[j]
            start_time_2 = task_start_times[s_id2]
            if np.abs(start_time - start_time_2) > delta:
                # if the difference between the start times is larger than delta, break the loop as following tasks start at a later time
                break
            swappable_steps[s_id].append(s_id2)
            swappable_steps[s_id2].append(s_id)
    # Remove empty keys
    for key in [key for key in swappable_steps if len(swappable_steps[key]) == 0]: del swappable_steps[key]
    # print(swappable_steps)
    return swappable_steps

def swap_close_tasks(schedule, team):
    swappable_steps = get_swappable_steps(schedule, team)
    if len(swappable_steps) == 0:
        return None
    new_schedule = schedule.copy()
    s1 = random.choice(list(swappable_steps.keys()))
    s2 = random.choice(swappable_steps[s1])
    tmp_task = new_schedule[s1][0]
    # print(new_schedule[s1], new_schedule[s2])
    new_schedule[s1][0] = new_schedule[s2][0]
    new_schedule[s2][0] = tmp_task
    return new_schedule

def random_gen_schedules(schedules, team, size_gen = 10):
    copy_schedules = copy.deepcopy(schedules)
    new_schedules = []
    # print(size_gen)
    for i in range(size_gen):
        s = random.choice(copy_schedules)
        new_schedule = swap_close_tasks(s.copy(), team)
        if new_schedule is not None:
            new_schedules.append(new_schedule)
    return new_schedules

def generate_evolution(baselines, count: int, elements=[0]):
    """[summary]

    Args:
        baselines (list of list): list of list of actions
        count (int): number of children produced
        elements (list, optional): [description]. Defaults to [0].

    Returns:
        [type]: [description]
    """
    # print(baseline)
    baselines_copy = copy.deepcopy(baselines)
    mutations = []
    for i in range(count):
        random_index = random.randint(0,len(baselines_copy)-1)
        baseline = copy.deepcopy(baselines_copy[random_index])
        new_model = np.array(baseline.copy(), dtype=int)
        swap_id = random.sample([i for i in range(len(baseline))], 2)
        # Select indices randomly
        # swap tasks
        tmp = new_model[swap_id[0]][elements] # Task_ID
        new_model[swap_id[0]][elements] = new_model[swap_id[1]][elements]
        new_model[swap_id[1]][elements] = tmp
        # print(new_model)
        mutations.append(new_model.tolist())
    # print(mutations)
    return mutations

def swap_task_allocation(baselines, count:int):
    return generate_evolution(baselines, count, elements=[0])

def swap_task_order(baselines, count:int):
    return generate_evolution(baselines, count, elements=[0, 1, 2])
        