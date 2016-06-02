#!/usr/bin/env python
"""
The Universe module defines classes for managing network representations
"""
__author__ = "Steven Munn"
__email__ = "sjmunn@umail.ucsb.edu"
__date__ = "05-23-2016"

# -------------------- Runtime communication
import logging

# -------------------- Linear Algebra and scientific computing
import numpy as np
import numpy.matlib
from scipy import sparse as sparse
from scipy.sparse import linalg as LA
from scipy import linalg as LAdense
from scipy import stats

from scipy.integrate import ode
from scipy.stats import norm
from scipy.stats import multivariate_normal
import numpy as np

# -------------------- NetworkX
import networkx as nx

# -------------------- Plotting and display libraries
import matplotlib as mpl
# matplotlib.use('Agg') # Use Agg b/c MacOSX backend bug
import matplotlib.pyplot as plt

# --------------------
"""
ODE integration function

                     [ w_00 w_01 ...         [ 1     M_0   M_0 ...
dM = [ 1 1 1 ...] x    w_10 w_11 ...    (x)    M_1   1     M_1 ...
                       w_20 w_21 ...]          M_2   M_2   1   ...]
dM = Ones x Weights (x) M_diag

Where (x) is term by term multiplication
and x is matrix multiplication
"""
def ODEfunction(t, M, Weights, nNodes):
    dM = np.ones( ( nNodes, 1))
    
    M_diag = np.matlib.repmat(M, nNodes,1)
    M_diag = np.transpose(M_diag)
    np.fill_diagonal(M_diag,1)

    W_x_M_diag = Weights * M_diag
    dM = np.dot( np.ones( ( 1, nNodes)), W_x_M_diag)

    return dM

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
        self.nNodes = self.Graph.number_of_nodes()
        self.WeightsMatrix = matrix
        self._nodeAnnotate()

    def removeNode(self, node):
        self.Graph.remove_node(node)
        self.nNodes = self.Graph.number_of_nodes()

    """
    Universe Simulation
    """        
    def simulateUniverse(self,time):
        # self.log.debug('Simulating ground truth data')
        tempText = 'For ' + str( self.nNodes) + ' nodes.'
        # self.log.debug(tempText)
        tempText = 'Over ' + str( np.size(time)) + ' time steps.'
        # self.log.debug(tempText)
        
        self.M = np.zeros(( self.nNodes, np.size(time)))
        # set initial conditions
        self.M[:,0] = np.ones(self.nNodes)
        M0 = np.ones((3,1))

        # Simulate noisless Universe
        r = ode( ODEfunction).set_integrator('dopri5')
        r.set_initial_value([M0[0], M0[1], M0[2]], time[0])
        r.set_f_params( self.WeightsMatrix, self.nNodes)
    
        # Run integration
        k = 1
        while r.successful() and k < np.size(time):
            r.integrate(r.t + 1)
            
            self.M[:,k] = r.y.ravel()
            k+=1

        # self.log.debug('Done simulating ground truth')
        # self.log.debug(self.M)

    """
    Class Initialization
    """
    def __init__(self):
        self.log=logging.getLogger("Flow_Net")
