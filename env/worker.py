"""
Created on Fri Sep  3 19:11:13 2021

@author: baltundas3

Schedule Worker Models based on benchmark/edfutils.py with modifications to allow for each worked to keep track of the relevant information.

"""

import math
import numpy as np
from enum import  Enum, auto

import sys
# setting path
sys.path.append('../')
from env.human_learning_rate import HumanLearningRate

class WorkerType(Enum):
    ROBOT = 0
    HUMAN = 1
    def __eq__(self, other):
        if type(self).__qualname__ != type(other).__qualname__:
            return NotImplemented
        return self.name == other.name and self.value == other.value

    def __hash__(self):
        return hash((type(self).__qualname__, self.name))

"""
Worker class
"""
class Worker(object):
    def __init__(self, w_id: int, w_type: WorkerType):
        """

        """
        self.id = w_id
        self.type = w_type
        self.schedule = []
        self.next_available_time = 0
        self.task_counter = {}
        
    def add_task(self, task_id):
        """adds a task to keep track.

        Args:
            task_id (int): identity of the task
        """
        if task_id in self.task_counter:
            self.task_counter[task_id] += 1
        else:
            self.task_counter[task_id] = 1
        
    def reset(self):
        pass

    def set_tasks(self, task_times):
        pass

    def get_duration_of_task(self, task_id):
        pass
    
"""
Robot class
"""
class Robot(Worker):
        
    def __init__(self, w_id):
        """
        
        """
        super(Robot, self).__init__(w_id, WorkerType.ROBOT)
        

    def set_tasks(self, task_times):
        self.tasks = task_times
    
    def get_duration_of_task(self, task_id):
        return self.tasks[task_id]
        

"""
Human class
The primary difference of the Human Class is that it learns based on different tasks being chosen
"""
class Human(Worker):
    def __init__(self, w_id, learning_rate : HumanLearningRate):
        """

        Args:
            w_id (int): worker ID
            func (function): a function for human learning based on the equation provided
        """
        super(Human, self).__init__(w_id, WorkerType.HUMAN)
        self.learning_rate = learning_rate
        # self.task_counter = {} # A dictionary containing the count for each task that the Human has done, to allow for modelling of the learning.

    def get_duration_of_task(self, task_id):
        count = 0 # default task count is 0 for tasks that the human has not done before
        if task_id in self.task_counter:
            count = self.task_counter[task_id]
        expected_time = self.learning_rate(count, task_id, self.id)
        return expected_time

    def reset(self):
        self.task_counter.clear() 

if __name__ == '__main__':
    f = HumanLearningRate("task_variables.txt")
    print(f.task_models)
    h = Human(0, f)
    print(h.get_duration_of_task(1))