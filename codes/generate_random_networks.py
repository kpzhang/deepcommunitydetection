# -*- coding: utf-8 -*-

from networkx.generators.community import random_partition_graph
import networkx as nx
import numpy as np

class RandNet():

    """
    This is the class to represent random network generation.

    """

    def __init__(self):

        super(RandNet, self).__init__()


    
    def generate_random_networks(n_nodes, n_community, p_in, P_out, seed=None, directed = False):
    """
        A method used to generate synthetic networks with communities. This method is based on the implementation of
        random_partition_graph() method in the networkx library.

        Parameters:
        ----------
        n_nodes : int
            the number of nodes in the network
        n_community : int
            the number of communities in the network
        p_in : float
            probability of edges with in groups
        p_out : float
            probability of edges between groups
        seed : int
            a seed for the random number generator
        directed: boolean
            whether to create a directed graph, default (False)
    """

    community_sizes = np.random.multinomial(n_nodes, np.ones(n_community)/n_community)
    G = random_partition_graph(community_sizes, p_in, P_out, seed, directed)

    with open('random_{}_{}_{}_{}_edge.txt'.format(n_nodes, n_community, p_in, P_out), 'w') as f:
        for edge in G.edges:
            f.write(str(edge[0])+' '+str(edge[1]) + '\n')
    with open('random_{}_{}_{}_{}_node.txt'.format(n_nodes, n_community, p_in, P_out), 'w') as f:
        for node in G.nodes:
            f.write(str(node) + '\n')