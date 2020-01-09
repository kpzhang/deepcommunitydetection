# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import os, sys
import time
import warnings
import numpy as np

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import intra_community_edges
from networkx.algorithms.community.quality import inter_community_non_edges
from networkx.algorithms.community.quality import performance
from networkx.algorithms.community import modularity

from load_data import gen_X, gen_A

import argparse

import networkx as nx

"""
This is the baseline modularity maximization (MM). It consists of the following steps:
	1. load data: nodes and edges files
	2. convert adjecency matrix into a graph in the format of networkX
	3. run MM to obtain the community assignment for all nodes
	4. output the performance evaluation

Inputs
----------
nodes : str
	each line is a node id
edges : str
	each line is a pair of nodes, separated by a space
eval : str
	whether to show the evaluation result
"""

np.random.seed(2020)

start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--nodes', default=None, help='The filename related to nodes')
parser.add_argument('--edges', default=None, help='The filename related to edges')
parser.add_argument('--eval', type=int, default='N', help='Showing the evaluation results or not')
args = parser.parse_args()


# 1. load data first
node_fname = args.nodes
edge_fname = args.edges
peval = args.eval
node_ids, X = gen_X(node_fname)  # get feature matrix X
normalized_laplacian_mx, adj_mx = gen_A(node_ids,edge_fname) # get the adjecency matrix A and do normalized laplacian transformtion

# 2. convert adjencency matrix to networkX graph
G = nx.from_numpy_matrix(np.matrix(adj_mx))

# 3. run modularity maximization algorithm to obtain community assignment
community_labels = list(greedy_modularity_communities(G))
for cid in range(len(community_labels)):
	members = sorted(community_labels[cid])
	for m in members:
		print(node_ids[m],',',cid)

end = time.time()
print("time consumed:",end-start)


# 4. Output the performance evaluation
#    The performance of a partition is the ratio of the number of intra-community edges plus inter-community non-edges with the total number of potential edges.
if peval.lower() == 'y':
	nlabels = len(set(community_labels))
	partition = []
	for i in range(nlabels):
		partition.append(set(community_labels[i]))

	intra_edges = intra_community_edges(G,partition)
	inter_edges = inter_community_non_edges(G,partition)
	n = len(G)
	total_pairs = n*(n-1)
	if not G.is_directed():
		total_pairs //= 2
	print('The performance:',(intra_edges + inter_edges)/total_pairs)
	print('# communities:',nlabels)
