# -*- coding: utf-8 -*-

import os, sys, re
import warnings

import numpy as np
from scipy.sparse.csgraph import laplacian

def gen_X(node_fname):
	""" Generates feature matrix X: nxm, where n is the number of nodes and m is the number of features.
		If the node has attribtues, then each line in the file: node_id <tab> a list of attributes separated
		spaces.
		If the node does not have attributes, then each line in the file is just a node_id.
	"""
	
	fh = open(node_fname,'r')
	lines = fh.readlines()
	fh.close()

	node_ids = []
	attrs = []
	for i in range(len(lines)):
		arr = lines[i].strip().split()
		node_ids.append(arr[0].strip())
		if len(arr) > 1:
			attrs.append([float(v) for v in arr[1:]])

	return (node_ids, np.array(attrs))


def gen_A(node_ids,edge_fname):
	""" Generates the normalized laplacian adjecency matrix D^-1/2*A'*D^-1/2: nxn, where A' = A + I. """

	n = len(node_ids)
	adj_mx = np.zeros((n,n))

	fh = open(edge_fname,'r')
	lines = fh.readlines()
	fh.close()

	for line in lines:
		arr = line.strip().split()
		node1 = arr[0].strip()
		node2 = arr[1].strip()
		i = node_ids.index(node1)
		j = node_ids.index(node2)
		adj_mx[i,j] = 1

		if len(arr) == 3:
			adj_mx[i,j] = int(arr[2].strip())

	adj_mx_pri = adj_mx + np.identity(n) # A' = A + I
	
	normalized_laplacian_mx = laplacian(adj_mx_pri, normed=False) # 

	return (normalized_laplacian_mx, adj_mx)