# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 20:08:45 2020

@author: pheno
"""

import numpy as np

import torch
from utils import build_hetgraph, hetgraph_node_helper, SchedulingEnv
from graph.hetgat import HeteroGATLayer, MultiHeteroGATLayer

# problem path
fname = 'gen/r2t20_001/00013'
solname = 'gen/r2t20_001v9/00013'

# initialize env
env = SchedulingEnv(fname)

# load solution
optimals = []
for i in range(env.num_robots):
    optimals.append(np.loadtxt(solname+'_%d.txt' % i, dtype=np.int32))

optimalw = np.loadtxt(solname+'_w.txt', dtype=np.int32)

for i in range(env.num_robots):
    print(optimals[i])

print(optimalw)

map_width = 2
# convert from int32 (np default) to int64
# torch uses int64 by default when converting from python int list
unsch_tasks = np.array(env.get_unscheduled_tasks(), dtype=np.int64)
valid_tasks = np.array(env.get_valid_tasks(0), dtype=np.int64)

hetg = build_hetgraph(env.halfDG, env.num_tasks, env.num_robots, env.dur,
                      map_width, np.array(env.loc, dtype=np.int64),
                      1.0, env.partials, unsch_tasks, 1, valid_tasks)

device = torch.device('cuda')

g = hetg.to(device)

# both are same
print(g.edges['use_time'].data['t'])
print(g['use_time'].edata['t'])

# this one is not correct
print(g.edata['use_time'])
# use this one instead, here etype should be full
print(g.edata['t'][('robot', 'use_time', 'task')])

'''
# draw the metagraph using graphviz
import pygraphviz as pgv
def plot_graph(nxg):
    ag = pgv.AGraph(strict=False, directed=True)
    for u, v, k in nxg.edges(keys=True):
        ag.add_edge(u, v, label=k)
    ag.layout('dot')
    ag.draw('./pics/metagraph.png')

plot_graph(g.metagraph())
'''

feat_dict = hetgraph_node_helper(env.halfDG.number_of_nodes(), env.partialw,
                                 env.partials, env.loc, env.dur, map_width,
                                 env.num_robots, len(valid_tasks))

in_dim = {'task': 6,
          'loc': 1,
          'robot': 1,
          'state': 4,
          'value': 1
          }

hid_dim = {'task': 64,
           'loc': 64,
           'robot': 64,
           'state': 64,
           'value': 64
           }

out_dim = {'task': 32,
          'loc': 32,
          'robot': 32,
          'state': 32,
          'value': 1
          }

cetypes = [('task', 'temporal', 'task'),
           ('task', 'located_in', 'loc'),('loc', 'near', 'loc'),
           ('task', 'assigned_to', 'robot'), ('robot', 'com', 'robot'),
           ('task', 'tin', 'state'), ('loc', 'lin', 'state'), 
           ('robot', 'rin', 'state'), ('state', 'sin', 'state'), 
           ('task', 'tto', 'value'), ('robot', 'rto', 'value'), 
           ('state', 'sto', 'value'), ('value', 'vto', 'value'),
           ('task', 'take_time', 'robot'), ('robot', 'use_time', 'task')]

# Test GAT layer
layer1 = HeteroGATLayer(in_dim, hid_dim, cetypes).to(device)
                
feat_dict_tensor = {}
for key in feat_dict:
    feat_dict_tensor[key] = torch.Tensor(feat_dict[key]).to(device) 

h1 = layer1(g, feat_dict_tensor)

# Test multi-head GAT layer
num_heads = 4

#layer2 = MultiHeteroGATLayer(hid_dim, out_dim, cetypes, num_heads).to(device)
layer2 = MultiHeteroGATLayer(hid_dim, out_dim, cetypes, num_heads, merge='avg').to(device)

h2 = layer2(g, h1)
print(h2['task'].shape)
print(h2['loc'].shape)
print(h2['robot'].shape)
print(h2['state'].shape)
print(h2['value'].shape)

# Test ScheduleNet as a whole
from hetnet import ScheduleNet4Layer

model = ScheduleNet4Layer(in_dim, hid_dim, out_dim, cetypes, num_heads).to(device)
model.eval()

results = model(g, feat_dict_tensor)

print(results['task'].shape)
print(results['loc'].shape)
print(results['robot'].shape)
print(results['state'].shape)
print(results['value'].shape)

print('Test passed.')