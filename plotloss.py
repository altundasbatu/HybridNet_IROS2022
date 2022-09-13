# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 18:23:30 2019

@author: pheno

plot loss from checkpoints
"""

import torch

import matplotlib.pyplot as plt
import numpy as np


fname = './new03/checkpoint_03960.tar'

cp = torch.load(fname)
loss = np.array(cp['loss'])

plt.figure(figsize=(25,5))
#plt.loglog(loss[-6000:])
plt.plot(loss[-500:])
#plt.ylim(0, 180)
plt.xlabel('no. of training steps')
#plt.xlim(1000,10000)
plt.ylabel('total loss')
#plt.title('Offset = 0.1')
plt.savefig('./new03/loss_03960.png')