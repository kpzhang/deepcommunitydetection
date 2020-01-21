# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import os,sys,re
import warnings
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GCN(nn.Module):
	"""
	A class used to represent a GCN
	

	Attributes
	----------
	nfeature : int
		the dimensionality of the input (attribute X.shape[1])
	nhidden1 : int
		the number of neuron units in the first hidden layer
	nhidden2 : int
		the number of neuron units in the second hidden layer
	nhidden3 : int
		the number of neuron units in the third hidden layer
	nclass : int
		the dimensionality of the output / the number of clusters
	bias : boolean
		whether to use the bias parameter in the network (default True)

	Methods
	--------
	forward(X, adj)
		This is a forward function taking attribute X and adjacency matrix A to obtain the output

	"""

	def __init__(self, nfeature, nhidden1, nhidden2, nhidden3, nclass, bias=True):

		super(GCN, self).__init__()

		self.weight1 = Parameter(torch.rand(nfeature,nhidden1), requires_grad=True)
		self.weight2 = Parameter(torch.rand(nhidden1,nhidden2), requires_grad=True)
		self.weight3 = Parameter(torch.rand(nhidden2,nhidden3), requires_grad=True)
		self.weight4 = Parameter(torch.rand(nhidden3,nclass), requires_grad=True)
		self.reset_parameter()



	def reset_parameter(self):
		
		np.random.seed(self.seed)
        	if self.seed is not None:
            		torch.manual_seed(self.seed)

		'''
		stdv1 = 1./math.sqrt(self.weight1.size(1))
		self.weight1.data.uniform_(-stdv1, stdv1)

		stdv2 = 1./math.sqrt(self.weight2.size(1))
		self.weight2.data.uniform_(-stdv2, stdv2)

		stdv3 = 1./math.sqrt(self.weight3.size(1))
		self.weight3.data.uniform_(-stdv3, stdv3)

		stdv4 = 1./math.sqrt(self.weight4.size(1))
		self.weight4.data.uniform_(-stdv4, stdv4)
		'''
		
		torch.nn.init.xavier_uniform(self.weight1,gain=nn.init.calculate_gain('tanh'))
        	torch.nn.init.xavier_uniform(self.weight2,gain=nn.init.calculate_gain('tanh'))
        	torch.nn.init.xavier_uniform(self.weight3,gain=nn.init.calculate_gain('tanh'))
		torch.nn.init.xavier_uniform(self.weight4,gain=nn.init.calculate_gain('tanh'))


	def forward(self, A, X):
		
		output_layer1 = F.tanh(A.mm(X).mm(self.weight1))
		output_layer2 = F.tanh(A.mm(output_layer1).mm(self.weight2))
		output_layer3 = F.tanh(A.mm(output_layer2).mm(self.weight3))
		output = F.log_softmax(A.mm(output_layer3).mm(self.weight4), dim=1)

		return output
