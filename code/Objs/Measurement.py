#!/usr/bin/env python
"""
The Measurement class simulates the time-evolution of the network
"""
# ================================================== Standard Packages
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

# ================================================== Local Definitions
"""
Measurement Class allows us to esimate the universe parameters
"""


class Measurement:
    """
    Simulate a measurement at a given node
    """

    def simulateNodeMeasurement(self, Universe, iNode, vv):
        self.vv = vv
        self.node = iNode
        M = Universe.M
        self.nTimeSteps = np.size(M[iNode, :])
        # Add noise of variance vv
        self.Mhat = M[iNode, :] + np.random.rand(1,
                                                 self.nTimeSteps) * np.sqrt(vv)
        self.Mhat = self.Mhat.ravel()

        self.log.debug('Simulated Measurement:')
        self.log.debug(self.Mhat)

    """
    Class Initialization
    """

    def __init__(self):
        self.log = logging.getLogger("Flow_Net")
