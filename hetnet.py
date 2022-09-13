# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 20:53:49 2021

@author: baltundas3

Heterogeneous Graph Attention Network for rollout policy

"""

#import torch
import torch.nn as nn

from graph.hetgat import MultiHeteroGATLayer
from env.hybrid_team import HybridTeam
from env.worker import WorkerType

# GNN - non-batched version
class HybridScheduleNet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, cetypes, num_heads=4):
        super(HybridScheduleNet, self).__init__()

        hid_dim_input = {}
        for key in hid_dim:
            hid_dim_input[key] = hid_dim[key] * num_heads
        # print(hid_dim)

        # # LSTM Layer
        # # self.LSTM_layer = nn.LSTM(input_size, hid_dim)
        # self.lstm_layer = nn.LSTM(1, hid_dim['worker'], num_heads)
        self.hidden = None
        # HetNet Layers
        self.layer1 = MultiHeteroGATLayer(in_dim, hid_dim, cetypes, num_heads)
        self.layer2 = MultiHeteroGATLayer(hid_dim_input, hid_dim, cetypes, 
                                          num_heads)
        self.layer3 = MultiHeteroGATLayer(hid_dim_input, out_dim, cetypes, 
                                          num_heads, merge='avg')

    def forward(self, g, feat_dict):
        """[summary]
        
            number of Q score nodes = number of available actions
        Args:
            g: DGL heterograph
            feat_dict: dictionary of input features
        Returns:
            [type]: [description]
        """
        h1 = self.layer1(g, feat_dict)
        h2 = self.layer2(g, h1)
        h3 = self.layer3(g, h2)
        
        return h3
        
# GNN - non-batched version - 4 layer
class HybridScheduleNet4Layer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, cetypes, num_heads=4):
        super(HybridScheduleNet4Layer, self).__init__()

        hid_dim_input = {}
        for key in hid_dim:
            hid_dim_input[key] = hid_dim[key] * num_heads
        
        self.layer1 = MultiHeteroGATLayer(in_dim, hid_dim, cetypes, num_heads)
        self.layer2 = MultiHeteroGATLayer(hid_dim_input, hid_dim, cetypes, 
                                          num_heads)
        self.layer3 = MultiHeteroGATLayer(hid_dim_input, hid_dim, cetypes, 
                                          num_heads)
        self.layer4 = MultiHeteroGATLayer(hid_dim_input, out_dim, cetypes, 
                                          num_heads, merge='avg')

        # TODO: add LSTM here


    '''
    input
        g: DGL heterograph
            number of Q score nodes = number of available actions
        feat_dict: dictionary of input features
    '''
    def forward(self, g, feat_dict):
        # TODO: One Hot Encoder and LSTM pass.

        h1 = self.layer1(g, feat_dict)
        h2 = self.layer2(g, h1)
        h3 = self.layer3(g, h2)
        h4 = self.layer4(g, h3)
        
        return h4  

