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
import numpy as np

# -------------------- NetworkX
import networkx as nx

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
        self.log.debug('Simulating ground truth data')
        tempText = 'For ' + str( self.nNodes) + ' nodes.'
        self.log.debug(tempText)
        tempText = 'Over ' + str( np.size(time)) + ' time steps.'
        self.log.debug(tempText)
        
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

        self.log.debug('Done simulating ground truth')
        self.log.debug(self.M)

    """
    Class Initialization
    """
    def __init__(self):
        self.log=logging.getLogger("Flow_Net")


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
        self.nTimeSteps = np.size( M[iNode,:])
        # Add noise of variance vv
        self.Mhat = M[iNode,:] + np.random.rand( 1, self.nTimeSteps)*np.sqrt( vv)
        self.Mhat = self.Mhat.ravel()

        self.log.debug('Simulated Measurement:')
        self.log.debug(self.Mhat)

    """
    Class Initialization
    """
    def __init__(self):
        self.log=logging.getLogger("Flow_Net")

"""
The estimate class computes parameters for a universe
from a measurement, or set of measurements
"""

class Estimate:
    """
    Draw a sample for a weight
    Uses similar notation to,
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.html
    """
    def _drawWeightSample(self, i, j):
        vk = self.thetas[( i, j)]
        xk = np.arange(np.size(vk))
        pk = self.paramPdfs[(i,j)]
        pk = pk/np.sum(pk)

        custm = stats.rv_discrete(name='custm', values=(xk, pk))

        randomIdx = custm.rvs(size=1)
        self.log.debug(vk)
        parameterVal = vk[int(randomIdx[0])]
        return parameterVal

    """
    Make a first guess as to the weight values
    """
    def _ogWeightsGuess(self):
        # We is the weight estimate
        self.We = np.zeros( ( self.iSzWeights, self.jSzWeights))
        for iWeight in range( 0, self.iSzWeights):
            for jWeight in range( 0, self.jSzWeights):
                self.We[iWeight, jWeight] = self._drawWeightSample( iWeight, jWeight)

    """
    Compute new Parameter Pdf
    """
    def _computeParamPdf(self, i, j, Measurement):
        thetaVals = self.thetas[(i, j)]

        prior = np.ones((self.nSteps,1))
        posterior_t = np.zeros((Measurement.nTimeSteps-1,self.nSteps))
        for iParamStep in range(0, self.nSteps):
            Wtrial = self.We
            Wtrial[i, j] = thetaVals[iParamStep]

            testVerse = Universe()
            testVerse.initFromMatrix(Wtrial)
            t = np.arange( Measurement.nTimeSteps)
            testVerse.simulateUniverse(t)

            posterior_t[0, iParamStep] = prior[iParamStep] * norm.pdf(Measurement.Mhat[1], testVerse.M[Measurement.node,1], Measurement.vv)
            for tStep in range(1,Measurement.nTimeSteps - 1):
                posterior_t[tStep, iParamStep] = posterior_t[tStep -1, iParamStep] * norm.pdf(Measurement.Mhat[tStep +1], testVerse.M[Measurement.node,tStep + 1], Measurement.vv)

        # We've learned nothing if the first posterior is beneath tolerance
        tol = 1e-1
        if (np.sum(posterior_t[0,:]) < tol):
            posterior_t[-1,:] = prior.ravel()
            
        for tStep in range(0,Measurement.nTimeSteps-1):
            if (np.sum(posterior_t[tStep,:])/float(self.nSteps)) != 0:
                posterior_t[tStep,:] = posterior_t[tStep,:] / (np.sum(posterior_t[tStep,:])/float(self.nSteps))
            else:
                posterior_t[tStep,:] = posterior_t[tStep - 1,:]

        #self.log.info('-1-Pdf for w_02')
        #self.log.info(posterior_t[-1,:])
        self.paramPdfs[(i, j)] = posterior_t[-1,:]
        
    """
    Perform a full step of Gibbs Sampling
    """
    def _gibbsSamplingStep(self, Measurement, iGStep):
        for iWeight in range( 0, self.iSzWeights):
            for jWeight in range( 0, self.jSzWeights):
                # self.log.info('For coords')
                # self.log.info( iWeight)
                # self.log.info( jWeight)
                # self.log.info( self.thetas[ iWeight, jWeight])
                isConstant = ( np.size(self.thetas[ iWeight, jWeight]) == 1)
                if isConstant != True:
                    self._computeParamPdf(iWeight, jWeight, Measurement)
                    self.We[iWeight, jWeight] = self._drawWeightSample( iWeight, jWeight)

    """
    Estimate Parameters
    """
    def estimateParameters(self, Universe, Measurement, Wl, Wu):
        # Parameter steps
        nSteps = 2000
        self.nSteps = nSteps
        # Gibbs steps
        self.gibbsSteps = 100
        
        nTimeSteps = Measurement.nTimeSteps
        self.iSzWeights = np.size( Wl, 0)
        self.jSzWeights = np.size( Wl, 1)

        self.paramPdfs = {} # Dictionary of prior distributions
        self.thetas = {} # Dictionary of parameter values
        for iWeight in range( 0, self.iSzWeights):
            for jWeight in range( 0, self.jSzWeights):
                if Wl[ iWeight, jWeight] == Wu[ iWeight, jWeight]:
                    self.paramPdfs[( iWeight, jWeight)] = np.ones((1, 1))
                    self.thetas[( iWeight, jWeight)] = np.array([ Wl[ iWeight, jWeight]])
                else:
                    wu = Wu[ iWeight, jWeight]
                    wl = Wl[ iWeight, jWeight]
                    stepSize = (wu - wl) / float(nSteps -1)

                    self.thetas[( iWeight, jWeight)] = np.arange( wl, wu + 0.5*stepSize, stepSize)
                    self.paramPdfs[( iWeight, jWeight)] = np.ones(( 1, nSteps))

        self._ogWeightsGuess()

        self.log.info('Original weight guess')
        self.log.info(self.We[:,:])
        for iGStep in range( 1, self.gibbsSteps):
            self._gibbsSamplingStep(Measurement, iGStep)
            self.log.info('New weights')
            self.log.info(self.We)
    """
    Class Initialization
    """
    def __init__(self):
        self.log=logging.getLogger("Flow_Net")
   
