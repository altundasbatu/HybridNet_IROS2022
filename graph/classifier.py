# -*- coding: utf-8 -*-
"""
Created on Fri October 14 12:47:42 2021

@author: baltundas

Policy Classifiers for Task and Action
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List

class Classifier(nn.Module):
    """Classifier Object
    """
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.saved_log_probs = []
        self.saved_entropy = []
        self.rewards = []

    def forward(self, input): #task_output, state_output, worker_output, value_output):
        """Forward Function of the Classifier used to generate the softmax of probabilities and

        Args:
            task_output (Tensor): [description]
            state_output (Tensor): [description]
            worker_output (Tensor): [description]
            value_output (Tensor): [description]

        Returns:
            x: Softmax Probability output of each index
        """
        # TODO: concatanate the outputs
        # print(input.shape)
        x = self.fc(input)
        # print(x.shape)
        return x.squeeze(dim=-1)

class ClassifierNonBias(nn.Module):
    """Classifier Object
    """
    def __init__(self, input_dim, output_dim):
        super(ClassifierNonBias, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.saved_log_probs = []
        self.saved_entropy = []
        self.rewards = []

    def forward(self, input): #task_output, state_output, worker_output, value_output):
        """Forward Function of the Classifier used to generate the softmax of probabilities and

        Args:
            task_output (Tensor): [description]
            state_output (Tensor): [description]
            worker_output (Tensor): [description]
            value_output (Tensor): [description]

        Returns:
            x: Softmax Probability output of each index
        """
        # TODO: concatanate the outputs
        # print(input.shape)
        x = self.fc(input)
        # print(x.shape)
        return x.squeeze(dim=-1)

class ClassifierDeep(nn.Module):
    """Classifier Object
    """
    def __init__(self, input_dim, output_dim):
        super(ClassifierDeep, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim * 32),
            nn.ReLU(),
            nn.Linear(output_dim * 32, output_dim * 32),
            nn.ReLU(),
            nn.Linear(output_dim * 32, output_dim)
        )
        self.saved_log_probs = []
        self.saved_entropy = []
        self.rewards = []

    def forward(self, input): #task_output, state_output, worker_output, value_output):
        """Forward Function of the Classifier used to generate the softmax of probabilities and

        Args:
            task_output (Tensor): [description]
            state_output (Tensor): [description]
            worker_output (Tensor): [description]
            value_output (Tensor): [description]

        Returns:
            x: Softmax Probability output of each index
        """
        # TODO: concatanate the outputs
        # print(input.shape)
        x = self.fc(input)
        # print(x.shape)
        return x.squeeze(dim=-1)


if __name__ == '__main__':
    # TODO
    m = nn.Conv1d(6, 12, 32, stride=32)
    input = torch.randn(1, 6, 32)
    output = m(input)
    print(output.shape)
    pass