"""
Created on Fri October 14 12:47:42 2021

@author: baltundas

Contains all the layers and handles all the control
"""

import numpy as np

import copy

import torch
import torch.nn as nn
from torch.distributions import Categorical
torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from hetnet import HybridScheduleNet
from hetnet import HybridScheduleNet4Layer

from graph.lstm_layer import LSTM_CellLayer, LSTM_CellLayer2
from graph.classifier import Classifier, ClassifierNonBias, ClassifierDeep

from utils import ReplayMemory, Transition, action_helper_rollout
from utils import hetgraph_node_helper, build_hetgraph

from lr_scheduler_train import fill_demo_data

from env.mrc_problem import MRCProblem
from env.hybrid_team import HybridTeam

from env.scheduling_env import SchedulingEnv

class HybridPolicyNet(nn.Module):
    def __init__(self, selection_mode: str = 'sample', detach_gap: int=10, pad_value: bool = True, task_filtering = True):
        """Hybrid Schedule Net

        Args:
            num_tasks (int, optional): [description]. Defaults to 10.
            num_robots (int, optional): [description]. Defaults to 10.
            num_humans (int, optional): [description]. Defaults to 2.
            detach_gap (int, optional): [description]. Defaults to 10.
            pad_value (bool, optional): [description]. Defaults to True.
        """
        super(HybridPolicyNet, self).__init__()

        self.selection_mode = selection_mode
        self.task_filtering = task_filtering
        
        self.detach_gap = 10
        self.pad_value = pad_value

        self.state_dim = 1 # x 32

        self.hidden_dim = 32

        self.worker_embedding_size = (self.state_dim + 1) * self.hidden_dim # worker + state = 64
        self.task_embedding_size = (self.state_dim + 1 + 1) * self.hidden_dim # state + worker_chosen + state
        self._init_gnn()
        self._init_lstm()
        self._init_classifiers()
        
    def _init_gnn(self):
        """ Initialize Graph Neural Network
        """
        in_dim = {'task': 6, # do not change
                #   'loc': 1,
                'worker': 1,
                'state': 4
                }

        hid_dim = {'task': 64,
                #    'loc': 64,
                'worker': 64,
                'human': 64,
                'state': 64
                }

        out_dim = {'task': 64, # (hx || cx)
                #    'loc': 64,
                'worker': 64, # (hx || cx)
                'state': 64 # (hx || cx)
                }

        cetypes = [('task', 'temporal', 'task'),
                #    ('task', 'located_in', 'loc'), ('loc', 'near', 'loc'),
                ('task', 'assigned_to', 'worker'), ('worker', 'com', 'worker'),
                ('task', 'tin', 'state'), # ('loc', 'lin', 'state'),
                ('worker', 'win', 'state'), ('state', 'sin', 'state'),
                ('task', 'take_time', 'worker'), ('worker', 'use_time', 'task')]
        
        self.gnn = HybridScheduleNet(in_dim, hid_dim, out_dim, cetypes).to(self.device)
        # num_heads = 8
        # self.gnn = ScheduleNet4Layer(in_dim, hid_dim, out_dim, cetypes, num_heads).to(self.device)
        
    def _init_lstm(self):
        # batch_size
        self.lstm_cell_task = LSTM_CellLayer(self.hidden_dim * 2, self.hidden_dim)
        self.lstm_cell_worker = LSTM_CellLayer(self.hidden_dim * 2, self.hidden_dim)
        self.lstm_cell_state = LSTM_CellLayer(self.hidden_dim * 2, self.hidden_dim)
        # self.lstm_cell_value = LSTM_CellLayer(self.value_hidden_dim, self.value_hidden_dim)

    def _init_classifiers(self):
        self.worker_classifier = Classifier(self.worker_embedding_size, 1)
        self.task_classifier = Classifier(self.task_embedding_size, 1)
        
        self.worker_classifier2 = Classifier(self.worker_embedding_size, 1)
        self.task_classifier2 = Classifier(self.task_embedding_size, 1)

    def select_worker(self, worker_out):
        worker_probs = F.softmax(worker_out, dim=-1)
        m = Categorical(worker_probs)
        worker_id = 0
        print(m.probs)
        if self.selection_mode == 'sample':
            worker_id = m.sample()
        elif self.selection_mode == 'argmax':
            worker_id = torch.argmax(m.probs)
        self.worker_classifier.saved_log_probs[-1] += m.log_prob(worker_id)
        return worker_id.item()

    def select_task(self, task_out, unscheduled_tasks):
        task_probs = F.softmax(task_out, dim=-1)
        m = Categorical(task_probs)
        idx = 0
        if self.selection_mode == 'sample':
            idx = m.sample()
        elif self.selection_mode == 'argmax':
            idx = torch.argmax(m.probs)
        self.task_classifier.saved_log_probs[-1] += m.log_prob(idx)
        task_id = unscheduled_tasks[idx.item()]
        return task_id # task_id is indexed from 1

    def merge_embeddings(self, task_embedding, state_embedding, num_task=None):
        """Merge Task Embeddings

        Args:
            task_embedding (torch.Tensor): task_num + 2 x 32
            state_embedding (torch.Tensor): 1x32
        
        Returns:
            merged_embedding: task_num + 2 x 64
        """
        # TODO: Check this
        state_embedding_ = state_embedding.repeat(num_task, 1)
        merged_embedding = torch.cat((task_embedding, state_embedding_), dim=1)
        # print(merged_embedding)
        return merged_embedding

    def filter_tasks(self, wait, unscheduled_tasks):
        unfiltered_task_set = set(unscheduled_tasks)
        to_filter = set([])
        for si, fj, dur in wait:
            # fj comes before si
            if fj in unfiltered_task_set and si not in to_filter:
                to_filter.add(si)
        filtered_tasks = list(unfiltered_task_set - to_filter)
        filtered_tasks.sort()
        return filtered_tasks
    
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
        g = g.to(self.device)
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

    def forward(self, env):
        """Generates an entire schedule

        Args:
            env(SingleRoundScheduler): Single-Round Scheduler Environment
            unscheduled_tasks(List[int]): List of unscheduled tasks
        Returns:
            schedule: A List of actions that contain the entire schedule for all unscheduled tasks passed
        """
        schedule = []
        
        env.reset() # Rset Single Round Scheduler
        g, feat_dict, unscheduled_tasks = self.get_variables(env)
        # Get the output embeddings from the GNN
        outputs = self.gnn(g, feat_dict)
        task_output = outputs['task'] # (hx || cx)
        state_output = outputs['state']
        worker_output= outputs['worker']
        
        num_tasks = len(unscheduled_tasks)
        # Initialize the Hidden States
        state_hx = state_output[:, :32]
        state_cx = state_output[:, 32:]
        state_output = state_hx
        state_hidden = (state_hx, state_cx) #self.lstm_cell_state.init_hidden(self.state_dim)
        # state_hidden = (state_output, state_hidden[1]) # hidden_state = state_output, hidden_cell = zeros

        worker_hx = worker_output[:, :32]
        worker_cx = worker_output[:, 32:]
        worker_output = worker_hx
        worker_hidden = (worker_hx, worker_cx) # self.lstm_cell_worker.init_hidden(len(env.team))
        # worker_hidden = (worker_output, worker_hidden[1])

        
        # task_hidden = self.lstm_cell_task.init_hidden(num_tasks)
        # remove s0 and f0 along with any scheduled
        task_output = task_output[unscheduled_tasks + 1]
        task_hx = task_output[:, :32]
        task_cx = task_output[:, 32:]
        task_output = task_hx
        task_hidden = (task_hx, task_cx) # (task_output, task_hidden[1])
        
        # # Take a step from the LSTM for the first step
        # # Update the State Embedding
        # # print(task_output[0], worker_output[0])
        # task_worker_embedding = torch.cat((task_output[0], worker_output[0]), dim=0).unsqueeze(0)
        # print(task_worker_embedding.shape)
        # state_output, state_hidden = self.lstm_cell_state(task_worker_embedding, state_hidden)
        
        # # Update the Task Embedding
        # chosen_task_embedding = task_worker_embedding.repeat(num_tasks, 1)
        # # print(chosen_task_embedding.shape, task_hidden[0].shape, task_hidden[1].shape)
        # task_output, task_hidden = self.lstm_cell_task(chosen_task_embedding, task_hidden)

        # # Update the Worker Embedding
        # chosen_worker_embedding = task_worker_embedding.repeat(len(env.team), 1)
        # worker_output, worker_hidden = self.lstm_cell_worker(chosen_worker_embedding, worker_hidden)

        # Create a new log probability 
        self.worker_classifier.saved_log_probs.append(0)
        self.task_classifier.saved_log_probs.append(0)
        
        first = True
        while len(unscheduled_tasks) > 0:
            feasible_tasks = unscheduled_tasks.copy()
            if self.task_filtering:
                feasible_tasks = self.filter_tasks(env.problem.wait, unscheduled_tasks)
            num_feasible = len(feasible_tasks)
            
            # Select Worker
            ## Get Worker Selection Embeddings
            worker_relevant_embedding = self.merge_embeddings(worker_output, state_output, len(env.team))

            ## Run Categorical
            worker_probs = None
            if first:
                worker_probs = self.worker_classifier(worker_relevant_embedding)
            else:
                worker_probs = self.worker_classifier2(worker_relevant_embedding)
            worker_id = self.select_worker(worker_probs)
            chosen_worker_embedding = worker_output[worker_id, :].unsqueeze(dim=0)

            if len(feasible_tasks) == 1:
                task_id = feasible_tasks[0]
                action = (task_id, worker_id, 1.0)
                schedule.append(action)
                index = np.where(unscheduled_tasks == task_id)
                index = index[0][0]
                unscheduled_tasks = np.delete(unscheduled_tasks, index)
                # print(schedule)
                continue

            # Select Task
            feasible_task_output = task_output.clone()
            if self.task_filtering:
                indices = torch.Tensor(np.in1d(np.array(unscheduled_tasks), np.array(feasible_tasks)).nonzero()).squeeze()
                # print(unscheduled_tasks, feasible_tasks, indices)
                # print(feasible_task_output)
                feasible_task_output = torch.index_select(feasible_task_output, 0, indices.to(torch.int64))
            ## Get Task Selection Embedding
            task_relevant_embedding = self.merge_embeddings(feasible_task_output, state_output, num_feasible)
            ## Integrate Worker Probability into the Task Selection
            task_relevant_embedding_ = self.merge_embeddings(task_relevant_embedding, chosen_worker_embedding, num_feasible)
            task_probs = None
            if first:
                task_probs = self.task_classifier(task_relevant_embedding_)
            else:
                task_probs = self.task_classifier2(task_relevant_embedding_)
            first = False
            ## Run Categorical
            task_id = self.select_task(task_probs, feasible_tasks)

            action = (task_id, worker_id, 1.0)
            schedule.append(action)
            # print(schedule)

            # Remove Chosen Task from Unscheduled Tasks and Hidden Dimension as in Kool et al 2019
            task_id = action[0]
            index = np.where(unscheduled_tasks == task_id)
            index = index[0][0]
            unscheduled_tasks = np.delete(unscheduled_tasks, index)
            num_tasks = num_tasks - 1

            # Remove the Chosen embedding from task_relevant_embedding
            chosen_task_embedding = task_output[index,:].unsqueeze(dim=0)
            task_output = torch.cat((task_hidden[0][:index,:], task_hidden[0][index+1:,:]))
            task_cell = torch.cat((task_hidden[1][:index,:], task_hidden[1][index+1:,:]))
            task_hidden = (task_output, task_cell)
            # Remove the Chosen task index from state
            
            # Take a step from the LSTM for the next step
            # Update the State Embedding
            task_worker_embedding = torch.cat((chosen_task_embedding, chosen_worker_embedding), dim=1)
            state_output, state_hidden = self.lstm_cell_state(task_worker_embedding, state_hidden)
            
            # Update the Task Embedding
            chosen_task_embedding = task_worker_embedding.repeat(num_tasks, 1)
            task_output, task_hidden = self.lstm_cell_task(chosen_task_embedding, task_hidden)

            # Update the Worker Embedding
            chosen_worker_embedding = task_worker_embedding.repeat(len(env.team), 1)
            worker_output, worker_hidden = self.lstm_cell_worker(chosen_worker_embedding, worker_hidden)

        return schedule
    
class HybridPolicyNetUpdateSelected(nn.Module):
    def __init__(self, 
                 selection_mode: str = 'sample', 
                 detach_gap: int=10, 
                 pad_value: bool = True, 
                 task_filtering = True,
                 verbose='none',
                 device = torch.device("cpu")):
        """Hybrid Schedule Net

        Args:
            num_tasks (int, optional): [description]. Defaults to 10.
            num_robots (int, optional): [description]. Defaults to 10.
            num_humans (int, optional): [description]. Defaults to 2.
            detach_gap (int, optional): [description]. Defaults to 10.
            pad_value (bool, optional): [description]. Defaults to True.
        """
        super(HybridPolicyNetUpdateSelected, self).__init__()
        self.device = device
        
        self.selection_mode = selection_mode
        self.task_filtering = task_filtering
        
        self.detach_gap = 10
        self.pad_value = pad_value

        self.verbose = verbose
        
        self.state_dim = 1 # x 32

        self.hidden_dim = 32

        self.worker_embedding_size = (self.state_dim + 1) * self.hidden_dim # worker + state = 64
        self.task_embedding_size = (self.state_dim + 1 + 1) * self.hidden_dim # state + worker_chosen + state
        self._init_gnn()
        self._init_lstm()
        self._init_classifiers()
        
    def _init_gnn(self):
        """ Initialize Graph Neural Network
        """
        in_dim = {'task': 6, # do not change
                #   'loc': 1,
                'worker': 1,
                'state': 4
                }

        hid_dim = {'task': 64,
                #    'loc': 64,
                'worker': 64,
                'human': 64,
                'state': 64
                }

        out_dim = {'task': 64, # (hx || cx)
                #    'loc': 64,
                'worker': 64, # (hx || cx)
                'state': 64 # (hx || cx)
                }

        cetypes = [('task', 'temporal', 'task'),
                #    ('task', 'located_in', 'loc'), ('loc', 'near', 'loc'),
                ('task', 'assigned_to', 'worker'), ('worker', 'com', 'worker'),
                ('task', 'tin', 'state'), # ('loc', 'lin', 'state'),
                ('worker', 'win', 'state'), ('state', 'sin', 'state'),
                ('task', 'take_time', 'worker'), ('worker', 'use_time', 'task')]
        
        self.gnn = HybridScheduleNet(in_dim, hid_dim, out_dim, cetypes).to(self.device)
        # num_heads = 8
        # self.gnn = ScheduleNet4Layer(in_dim, hid_dim, out_dim, cetypes, num_heads).to(self.device)
        
    def _init_lstm(self):
        # batch_size
        self.lstm_cell_task = LSTM_CellLayer(self.hidden_dim * 2, self.hidden_dim)
        self.lstm_cell_worker = LSTM_CellLayer(self.hidden_dim * 2, self.hidden_dim)
        self.lstm_cell_state = LSTM_CellLayer(self.hidden_dim * 2, self.hidden_dim)
        # self.lstm_cell_value = LSTM_CellLayer(self.value_hidden_dim, self.value_hidden_dim)

    def _init_classifiers(self):
        self.worker_classifier = Classifier(self.worker_embedding_size, 1)
        self.task_classifier = Classifier(self.task_embedding_size, 1)
        
        # self.worker_classifier2 = Classifier(self.worker_embedding_size, 1)
        # self.task_classifier2 = Classifier(self.task_embedding_size, 1)

    def select_worker(self, worker_out):
        worker_probs = F.softmax(worker_out, dim=-1)
        m = Categorical(worker_probs)
        worker_id = 0
        if self.verbose == 'worker' or self.verbose == 'all':
            print(m.probs)
        if self.selection_mode == 'sample':
            worker_id = m.sample()
        elif self.selection_mode == 'argmax':
            worker_id = torch.argmax(m.probs)
        self.worker_classifier.saved_log_probs[-1] += m.log_prob(worker_id)
        self.worker_classifier.saved_entropy[-1] += m.entropy().mean() / self.num_workers # num_tasks # self.num_workers
        # print(m.log_prob(worker_id), m.entropy(), m.entropy().mean())
        return worker_id.item()

    def select_task(self, task_out, unscheduled_tasks):
        task_probs = F.softmax(task_out, dim=-1)
        m = Categorical(task_probs)
        idx = 0
        if self.verbose == 'task' or self.verbose == 'all':
            print(m.probs)
        if self.selection_mode == 'sample':
            idx = m.sample()
        elif self.selection_mode == 'argmax':
            idx = torch.argmax(m.probs)
        self.task_classifier.saved_log_probs[-1] += m.log_prob(idx)
        self.task_classifier.saved_entropy[-1] += m.entropy().mean() / self.num_tasks
        task_id = unscheduled_tasks[idx.item()]
        return task_id # task_id is indexed from 1

    def merge_embeddings(self, task_embedding, state_embedding, num_task=None):
        """Merge Task Embeddings

        Args:
            task_embedding (torch.Tensor): task_num + 2 x 32
            state_embedding (torch.Tensor): 1x32
        
        Returns:
            merged_embedding: task_num + 2 x 64
        """
        # TODO: Check this
        state_embedding_ = state_embedding.repeat(num_task, 1)
        merged_embedding = torch.cat((task_embedding, state_embedding_), dim=1)
        # print(merged_embedding)
        return merged_embedding

    def filter_tasks(self, wait, unscheduled_tasks):
        unfiltered_task_set = set(unscheduled_tasks)
        to_filter = set([])
        for si, fj, dur in wait:
            # fj comes before si
            if fj in unfiltered_task_set and si not in to_filter:
                to_filter.add(si)
        filtered_tasks = list(unfiltered_task_set - to_filter)
        filtered_tasks.sort()
        return filtered_tasks
    
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
        g = g.to(self.device)
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
            feat_dict_tensor[key] = torch.Tensor(feat_dict[key]).to(self.device)
        
        return g, feat_dict_tensor, unsch_tasks

    def forward(self, env: SchedulingEnv):
        """Generates an entire schedule

        Args:
            env(SingleRoundScheduler): Single-Round Scheduler Environment
            unscheduled_tasks(List[int]): List of unscheduled tasks
        Returns:
            schedule: A List of actions that contain the entire schedule for all unscheduled tasks passed
        """
        schedule = []
        
        env.reset() # Rset Single Round Scheduler
        g, feat_dict, unscheduled_tasks = self.get_variables(env)
        # Get the output embeddings from the GNN
        outputs = self.gnn(g, feat_dict)
        task_output_raw = outputs['task'] # (hx || cx)
        state_output_raw = outputs['state']
        worker_output_raw = outputs['worker']
        
        self.num_tasks = len(unscheduled_tasks)
        self.num_workers = len(env.team)
        # Initialize the Hidden States
        state_hx = state_output_raw[:, :32]
        state_cx = state_output_raw[:, 32:]
        state_output = state_hx
        state_hidden = (state_hx, state_cx) #self.lstm_cell_state.init_hidden(self.state_dim)

        worker_hx = worker_output_raw[:, :32]
        worker_cx = worker_output_raw[:, 32:]
        worker_output = worker_hx
        worker_hidden = (worker_hx, worker_cx) # self.lstm_cell_worker.init_hidden(len(env.team))

        # remove s0 and f0 along with any scheduled
        task_output_raw_ = task_output_raw[unscheduled_tasks + 1]
        task_hx = task_output_raw_[:, :32]
        task_cx = task_output_raw_[:, 32:]
        task_output = task_hx
        task_hidden = (task_hx, task_cx) # (task_output, task_hidden[1])

        # Create a new log probability and entropy
        ## Worker Classifier 
        self.worker_classifier.saved_log_probs.append(0)
        self.worker_classifier.saved_entropy.append(0)
        ## Task Classifier:
        self.task_classifier.saved_log_probs.append(0)
        self.task_classifier.saved_entropy.append(0)
        
        while len(unscheduled_tasks) > 0:
            feasible_tasks = unscheduled_tasks.copy()
            if self.task_filtering:
                feasible_tasks = self.filter_tasks(env.problem.wait, unscheduled_tasks)
            num_feasible = len(feasible_tasks)
            
            # Select Worker
            ## Get Worker Selection Embeddings
            worker_relevant_embedding = self.merge_embeddings(worker_output, state_output, len(env.team))

            ## Run Categorical
            worker_probs = self.worker_classifier(worker_relevant_embedding)
            worker_id = self.select_worker(worker_probs)
            chosen_worker_embedding = worker_output[worker_id, :].unsqueeze(dim=0)

            if len(feasible_tasks) == 1:
                task_id = feasible_tasks[0]
                action = [task_id, worker_id, 1.0]
                schedule.append(action)
                index = np.where(unscheduled_tasks == task_id)
                index = index[0][0]
                unscheduled_tasks = np.delete(unscheduled_tasks, index)
                # print(schedule)
                continue

            # Select Task
            if self.task_filtering:
                feasible_task_output_ = task_output.clone()
                possible_tasks = np.in1d(np.array(unscheduled_tasks), np.array(feasible_tasks)).nonzero()[0]
                indices = torch.tensor(np.array(possible_tasks), device=self.device)
                # print(unscheduled_tasks, feasible_tasks, indices)
                # print(feasible_task_output)
                feasible_task_output = torch.index_select(feasible_task_output_, 0, indices.to(self.device, torch.int64))
            else:
                feasible_task_output = task_output.clone()
            ## Get Task Selection Embedding
            task_relevant_embedding = self.merge_embeddings(feasible_task_output, state_output, num_feasible)
            ## Integrate Worker Probability into the Task Selection
            task_relevant_embedding_ = self.merge_embeddings(task_relevant_embedding, chosen_worker_embedding, num_feasible)
            task_probs = self.task_classifier(task_relevant_embedding_)
            ## Run Categorical
            task_id = self.select_task(task_probs, feasible_tasks)

            action = [task_id, worker_id, 1.0]
            schedule.append(action)
            # print(schedule)

            # Remove Chosen Task from Unscheduled Tasks and Hidden Dimension as in Kool et al 2019
            task_id = action[0]
            index = np.where(unscheduled_tasks == task_id)
            index = index[0][0]
            unscheduled_tasks = np.delete(unscheduled_tasks, index)
            num_tasks = len(unscheduled_tasks)

            # Remove the Chosen embedding from task_relevant_embedding
            chosen_task_embedding = task_output[index,:].unsqueeze(dim=0)
            task_output = torch.cat((task_hidden[0][:index,:], task_hidden[0][index+1:,:]))
            task_cell = torch.cat((task_hidden[1][:index,:], task_hidden[1][index+1:,:]))
            task_hidden = (task_output, task_cell)
            
            # Take a step from the LSTM for the next step
            # Update the State Embedding
            task_worker_embedding = torch.cat((chosen_task_embedding, chosen_worker_embedding), dim=1)
            state_output, state_hidden = self.lstm_cell_state(task_worker_embedding, state_hidden)
            
            # # Update the Task Embedding
            # chosen_task_embedding = task_worker_embedding.repeat(num_tasks, 1)
            # task_output, task_hidden = self.lstm_cell_task(chosen_task_embedding, task_hidden)

            # Update the Selected Worker Embedding
            # print(chosen_worker_embedding.shape)
            # print(worker_hidden[0].shape, worker_hidden[1].shape)
            chosen_worker_hidden = (worker_hidden[0][worker_id].unsqueeze(dim=0), worker_hidden[1][worker_id].unsqueeze(dim=0))
            worker_output_replacement, chosen_worker_hidden = self.lstm_cell_worker(task_worker_embedding, chosen_worker_hidden)
            # Prevent Inplace replacement error in Pytorch by using the following to replace single embedding
            worker_output_tmp = torch.cat((worker_output[:worker_id], worker_output_replacement, worker_output[worker_id + 1:]), dim=0)
            worker_output = worker_output_tmp
        return schedule

class HybridPolicyNetNonRecursive(nn.Module):
    def __init__(self, 
                 selection_mode: str = 'sample', 
                 detach_gap: int=10, 
                 pad_value: bool = True, 
                 task_filtering = True,
                 verbose='none',
                 device = torch.device("cpu")):
        """Hybrid Schedule Net

        Args:
            num_tasks (int, optional): [description]. Defaults to 10.
            num_robots (int, optional): [description]. Defaults to 10.
            num_humans (int, optional): [description]. Defaults to 2.
            detach_gap (int, optional): [description]. Defaults to 10.
            pad_value (bool, optional): [description]. Defaults to True.
        """
        super(HybridPolicyNetNonRecursive, self).__init__()

        self.device = device
        self.selection_mode = selection_mode
        self.task_filtering = task_filtering
        
        self.pad_value = pad_value

        self.verbose = verbose

        self.state_dim = 1 # x 32

        self.hidden_dim = 32

        self.worker_embedding_size = (self.state_dim + 1) * self.hidden_dim # worker + state = 64
        self.task_embedding_size = (self.state_dim + 1 + 1) * self.hidden_dim # state + worker_chosen + state
        self._init_gnn()
        self._init_classifiers()
        
    def _init_gnn(self):
        in_dim = {'task': 6, # do not change
                #   'loc': 1,
                'worker': 1,
                'state': 4
                }

        hid_dim = {'task': 64,
                #    'loc': 64,
                'worker': 64,
                'human': 64,
                'state': 64
                }

        out_dim = {'task': 32,
                #    'loc': 32,
                'worker': 32,
                'state': 32
                }

        cetypes = [('task', 'temporal', 'task'),
                #    ('task', 'located_in', 'loc'), ('loc', 'near', 'loc'),
                ('task', 'assigned_to', 'worker'), ('worker', 'com', 'worker'),
                ('task', 'tin', 'state'), # ('loc', 'lin', 'state'),
                ('worker', 'win', 'state'), ('state', 'sin', 'state'),
                ('task', 'take_time', 'worker'), ('worker', 'use_time', 'task')]
        
        self.gnn = HybridScheduleNet(in_dim, hid_dim, out_dim, cetypes).to(self.device)
        # num_heads = 8
        # self.gnn = ScheduleNet4Layer(in_dim, hid_dim, out_dim, cetypes, num_heads).to(self.device)
        
    def _init_classifiers(self):
        self.worker_classifier = ClassifierDeep(self.worker_embedding_size, 1)
        self.task_classifier = ClassifierDeep(self.task_embedding_size, 1)

    def select_worker(self, worker_out):
        worker_probs = F.softmax(worker_out, dim=-1)
        m = Categorical(worker_probs)
        worker_id = 0
        if self.verbose == 'worker' or self.verbose == 'all':
            print(m.probs)
        if self.selection_mode == 'sample':
            worker_id = m.sample()
        elif self.selection_mode == 'argmax':
            worker_id = torch.argmax(m.probs)
        self.worker_classifier.saved_log_probs[-1] += m.log_prob(worker_id)
        self.worker_classifier.saved_entropy[-1] += m.entropy().mean() / self.num_tasks # / num_workers
        return worker_id.item()

    def select_task(self, task_out, unscheduled_tasks):
        task_probs = F.softmax(task_out, dim=-1)
        m = Categorical(task_probs)
        idx = 0
        if self.verbose == 'task' or self.verbose == 'all':
            print(m.probs)
        if self.selection_mode == 'sample':
            idx = m.sample()
        elif self.selection_mode == 'argmax':
            idx = torch.argmax(m.probs)
        self.task_classifier.saved_log_probs[-1] += m.log_prob(idx)
        self.task_classifier.saved_entropy[-1] += m.entropy().mean() / self.num_tasks
        task_id = unscheduled_tasks[idx.item()]
        return task_id # task_id is indexed from 1

    def merge_embeddings(self, task_embedding, state_embedding, num_task=None):
        """Merge Task Embeddings

        Args:
            task_embedding (torch.Tensor): task_num + 2 x 32
            state_embedding (torch.Tensor): 1x32
        
        Returns:
            merged_embedding: task_num + 2 x 64
        """
        state_embedding_ = state_embedding.repeat(num_task, 1)
        merged_embedding = torch.cat((task_embedding, state_embedding_), dim=1)
        return merged_embedding

    def filter_tasks(self, wait, unscheduled_tasks):
        unfiltered_task_set = set(unscheduled_tasks)
        to_filter = set([])
        for si, fj, dur in wait:
            # fj comes before si
            if fj in unfiltered_task_set and si not in to_filter:
                to_filter.add(si)
        filtered_tasks = list(unfiltered_task_set - to_filter)
        filtered_tasks.sort()
        return filtered_tasks
    
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
        g = g.to(self.device)
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
            feat_dict_tensor[key] = torch.Tensor(feat_dict[key]).to(self.device)
        
        return g, feat_dict_tensor, unsch_tasks

    def forward(self, env):
        """Generates an entire schedule

        Args:
            g (DGL): [description]
            feat_dict (Tensor): [description]
            unscheduled_tasks(List[int]): List of unscheduled tasks
        Returns:
            schedule: A List of actions that contain the entire schedule for all unscheduled tasks passed
        """
        schedule = []
        
        env.reset() # Reset Single Round Scheduler
        self.num_tasks = env.problem.num_tasks
        # Create a new log probability and entropy
        ## Worker Classifier 
        self.worker_classifier.saved_log_probs.append(0)
        self.worker_classifier.saved_entropy.append(0)
        ## Task Classifier:
        self.task_classifier.saved_log_probs.append(0)
        self.task_classifier.saved_entropy.append(0)
        g, feat_dict, unscheduled_tasks = self.get_variables(env)
        
        while len(unscheduled_tasks) > 0:
            # Get the output embeddings from the GNN
            outputs = self.gnn(g, feat_dict)
            task_output = outputs['task']
            state_output = outputs['state']
            worker_output= outputs['worker']
            
            num_tasks = len(unscheduled_tasks)
            
            # remove s0 and f0 along with any scheduled
            task_output = task_output[unscheduled_tasks + 1]
            feasible_tasks = unscheduled_tasks.copy()
            if self.task_filtering:
                feasible_tasks = self.filter_tasks(env.problem.wait, unscheduled_tasks)
            num_feasible = len(feasible_tasks)
            
            # Select Worker
            ## Get Worker Selection Embeddings
            worker_relevant_embedding = self.merge_embeddings(worker_output, state_output, len(env.team))

            ## Run Categorical
            worker_probs = self.worker_classifier(worker_relevant_embedding)
            worker_id = self.select_worker(worker_probs)
            chosen_worker_embedding = worker_output[worker_id, :].unsqueeze(dim=0)

            if len(feasible_tasks) == 1:
                task_id = feasible_tasks[0]
                action = (task_id, worker_id, 1.0)
                schedule.append(action)
                index = np.where(unscheduled_tasks == task_id)
                index = index[0][0]
                unscheduled_tasks = np.delete(unscheduled_tasks, index)
                # print(schedule)
                continue

            # Select Task
            if self.task_filtering:
                feasible_task_output_ = task_output.clone()
                possible_tasks = np.in1d(np.array(unscheduled_tasks), np.array(feasible_tasks)).nonzero()[0]
                indices = torch.tensor(np.array(possible_tasks), device=self.device)
                # print(unscheduled_tasks, feasible_tasks, indices)
                # print(feasible_task_output)
                feasible_task_output = torch.index_select(feasible_task_output_, 0, indices.to(self.device, torch.int64))
            else:
                feasible_task_output = task_output.clone()
            ## Get Task Selection Embedding
            task_relevant_embedding = self.merge_embeddings(feasible_task_output, state_output, num_feasible)
            ## Integrate Worker Probability into the Task Selection
            task_relevant_embedding_ = self.merge_embeddings(task_relevant_embedding, chosen_worker_embedding, num_feasible)
            task_probs = self.task_classifier(task_relevant_embedding_)
            ## Run Categorical
            task_id = self.select_task(task_probs, feasible_tasks)

            action = [task_id, worker_id, 1.0]
            schedule.append(action)
            # print(schedule)

            # Remove Chosen Task from Unscheduled Tasks and Hidden Dimension as in Kool et al 2019
            task_id = action[0]
            index = np.where(unscheduled_tasks == task_id)
            index = index[0][0]
            unscheduled_tasks = np.delete(unscheduled_tasks, index)
            num_tasks = num_tasks - 1
            
            env.step(action[0], action[1], action[2])
            g, feat_dict, unscheduled_tasks = self.get_variables(env)

        return schedule

if __name__ == '__main__':

    num_tasks = 10
    num_robots = 5
    num_humans = 2

    i = 1
    # Memory and Environments
    folder = 'tmp/small_training_set'
    problem_file_name = folder + "/problems/problem_" +  format(i, '04')
    problem = MRCProblem(fname = problem_file_name)
    team = HybridTeam(problem)
    env = SchedulingEnv(problem=problem, team=team)
    g, feat_dict_tensor, unsch_tasks = get_variables(env)
    
    # Model
    model = HybridPolicyNet()

    # Output for Schedule
    output = model(g, feat_dict_tensor, env, unsch_tasks)
    print(output)


        