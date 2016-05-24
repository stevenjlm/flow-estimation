#!/usr/bin/env python
"""
The Universe module defines classes for managing network representations
"""
__author__ = "Steven Munn"
__email__ = "sjmunn@umail.ucsb.edu"
__date__ = "05-23-2016"
# Dependancies
# Runtime communication
import logging

# # Linear Algebra and scientific computing
# import numpy as np
# from scipy import sparse as sparse
# from scipy.sparse import linalg as LA
# from scipy import linalg as LAdense

# import copy

# NetworkX
import networkx as nx

"""
The universe class describes the graph which we are studying
"""

class Universe:

    """
    Graph Attribute Management
    """
    def _nodeAnnotate (self):
        for n,d in self.Graph.nodes_iter(data=True):
            self.Graph.node[n]['number'] = n

    def signal2Attributes (self,signal,signalName='signal'):
        signalNorm=signal # /float(np.amax(signal))
        for n,d in self.Graph.nodes_iter(data=True):
            #assert len(signalNorm[n]) == 1, \
            #    'Signal Norm is longer than one'
            self.Graph.node[n][signalName] = np.asscalar(signalNorm[n])

    """
    Graph Structure Management
    """
    def initRandom(self,innNodes):
        # Construct the graph randomly
        self.nNodes=innNodes
        self.Graph=nx.DiGraph()
        self.Graph=nx.powerlaw_cluster_graph(self.nNodes,2,.3)
        self._nodeAnnotate()

    def initFromData(self,edges):
        self.Graph=nx.DiGraph()
        self.Graph.add_edges_from(edges)
        self.nNodes=self.Graph.number_of_nodes()
        self._nodeAnnotate()

    def initFromMatrix(self,matrix):
        self.Graph = nx.DiGraph()
        self.Graph = nx.from_numpy_matrix(matrix)
        self._nodeAnnotate()

    def removeNode(self, node):
        self.Graph.remove_node(node)
        self.nNodes=self.Graph.number_of_nodes()

    """
    Universe Simulation
    """
    def simulateUniverse(self,time):
        
        
    """
    Class Initialization
    """
    def __init__(self):
        self.log=logging.getLogger("Flow_Net")
