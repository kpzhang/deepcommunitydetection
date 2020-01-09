# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

from networkx.algorithms.community.quality import intra_community_edges
from networkx.algorithms.community.quality import inter_community_non_edges
from networkx.algorithms.community import modularity
from networkx.algorithms.community.quality import performance

import os,sys
import time
import warnings
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import GCN

from load_data import gen_X, gen_A

import argparse

"""
This is the main function to start. It consists of the following steps:
	1. load data: nodes and edges files
	2. convert adjecency matrix into a graph in the format of networkX
	3. run GCN to obtain the embedding vectors for nodes
	4. print out the community assignment for all nodes
	5. output the performance evaluation

Inputs
----------
nodes : str
	each line is a node id
edges : str
	each line is a pair of nodes, separated by a space
node attributes : str
	whether nodes have attributes or not
hidden1 : int
	the number neurons in the first hidden layer
hidden2 : int
	the number neurons in the second hidden layer
hidden3 : int
	the number neurons in the third hidden layer
nlabels : int
	the number of clusters
eval : str
	whether to show the evaluation result
"""

os.environ['KMP_DUPLICATE_LIB_OK']='True'

np.random.seed(2020)

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--nodes', default=None, help='The filename related to nodes')
parser.add_argument('--edges', default=None, help='The filename related to edges')
parser.add_argument('--nodeattr', default='Y', help='Having node attributes')
parser.add_argument('--hidden1', type=int, default=128, help='The number of first hidden layer units')
parser.add_argument('--hidden2', type=int, default=64, help='The number of second hidden layer units')
parser.add_argument('--hidden3', type=int, default=128, help='The number of third hidden layer units')
parser.add_argument('--nlabels', type=int, default=100, help='The number of output units')
parser.add_argument('--eval', type=int, default='N', help='Showing the evaluation results or not')
args = parser.parse_args()

# 1. load data first
node_fname = args.nodes
edge_fname = args.edges
node_attr = args.nodeattr
peval = args.eval
node_ids, X = gen_X(node_fname)  # get feature matrix X
normalized_laplacian_mx, adj_mx = gen_A(node_ids,edge_fname) # get the adjecency matrix A and do normalized laplacian transformtion

# 2. convert adjencency matrix to networkX graph
G = nx.from_numpy_matrix(np.matrix(adj_mx))


# 3. run GCN
if len(X) == 0 or node_attr.lower() == 'n':
	X = np.identity(len(node_ids))
model = GCN(nfeature=X.shape[1], nhidden1=args.hidden1, nhidden2=args.hidden2, nhidden3=args.hidden3, nclass=args.nlabels)
X = torch.FloatTensor(X)
normalized_laplacian_mx = torch.FloatTensor(normalized_laplacian_mx)

model.train()
output = model(normalized_laplacian_mx, X)


end = time.time()
print("time consumed:",end-start)


# 4. print community assignment: nodeid, communityid
output = model(normalized_laplacian_mx,X)
community_labels = np.argmax(output.detach().numpy(),axis=1).tolist()
partition = [set() for k in range(args.nlabels)]
for i in range(len(community_labels)):
	#print(node_ids[i],',',community_labels[i])
	partition[community_labels[i]].add(i)

# 5. Output the performance evaluation
# 	 The performance of a partition is the ratio of the number of intra-community edges plus inter-community non-edges with the total number of potential edges.
if peval.lower() == 'y':
	intra_edges = intra_community_edges(G,partition)
	inter_edges = inter_community_non_edges(G,partition)
	n = len(G)
	total_pairs = n*(n-1)
	if not G.is_directed():
    	total_pairs //= 2
	print('The performance:',(intra_edges + inter_edges)/total_pairs)
