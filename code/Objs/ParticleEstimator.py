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

# -------------------- Local
from Universe import *
# --------------------


"""
Particle Filter Estimator
"""
class ParticleEstimator:
    """
    Draw a sample for a weight
    """
    def _drawOgWeightSample(self, i, j):
        s = np.random.uniform(self.Wl[i,j], self.Wu[i,j],1)
        return s[0]

    """
    Make a first guess as to the weight values
    """
    def _ogWeightsGuess(self, iParticle):
        # We is the weight estimate
        for iWeight in range( 0, self.iSzWeights):
            for jWeight in range( 0, self.jSzWeights):
                self.We[iWeight, jWeight, iParticle] = self._drawOgWeightSample( iWeight, jWeight)

    """
    Compute importance weights
    """
    def _computeImportanceWeights(self, Measurement, tStep):
        for iParticle in range(0, self.nParticles):
            Wtrial = self.We[:,:,iParticle]

            testVerse = Universe()
            testVerse.initFromMatrix(Wtrial)
            t = np.arange( tStep + 1)
            testVerse.simulateUniverse(t)

            self.log.debug('Measurement')
            self.log.debug( Measurement.Mhat[ tStep])
            self.log.debug('test val')
            self.log.debug( testVerse.M[ Measurement.node, tStep])
            iw = norm.pdf( Measurement.Mhat[ tStep], testVerse.M[ Measurement.node, tStep], Measurement.vv)
            self.importanceWeights[iParticle] = iw

    """
    Draw a sample particle number
    Uses similar notation to,
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.html
    """
    def _drawParticleNumber(self):
        xk = np.arange( self.nParticles)
        pk = self.importanceWeights
        pk = pk/np.sum(pk)

        custm = stats.rv_discrete(name='custm', values=(xk, pk))

        randomIdx = custm.rvs(size=1)
        return randomIdx

    """
    Resample
    """
    def _resample(self, tStep):
        graphWeightsBar = np.zeros( ( self.iSzWeights, self.jSzWeights, self.nParticles))
        for iParticle in range(0, self.nParticles):
            particleIdx = self._drawParticleNumber()
            graphWeightsBar[:,:,iParticle] = np.squeeze(self.We[:,:,particleIdx])

        self.Wbar[tStep] = graphWeightsBar

    """
    We Histogram plots
    """
    def plotGraphWeightsDistribution(self, i, j, tStep):
        plt.hist(self.We[i,j,:],100)
        plt.title("Graph Weights distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        
        filename = "out/" + str(tStep) + "_WeDistr.pdf"
        plt.savefig(filename,transparent=True) # save as pdf for higher quality
        plt.close()

    """
    Estimate Parameters using a particle filter
    """
    def estimateParameters(self, Universe, Measurement, Wl, Wu):
        self.Wu = Wu
        self.Wl = Wl
        self.iSzWeights = np.size( Wl, 0)
        self.jSzWeights = np.size( Wl, 1)
        self.nTimeSteps = Measurement.nTimeSteps
        
        self.nParticles = 10000
        self.We = np.zeros( ( self.iSzWeights, self.jSzWeights, self.nParticles))
        self.Wbar = {}

        # Initialize Particle filter
        for iParticle in range(0, self.nParticles):
            self._ogWeightsGuess( iParticle)

        """
        ==============================
        DEBUG
        ==============================
        """
        self.We[:,:,0] = np.array([[1, 0, 1],
                                   [0, 0, 1],
                                   [0, -1.45, 0]])
        # ============================
        self.log.debug('Originial Weights')
        for iParticle in range(0, self.nParticles):
            self.log.debug(self.We[:,:,iParticle])

        tStep = 1
        self.importanceWeights = np.zeros( ( self.nParticles, 1))
        self._computeImportanceWeights( Measurement, tStep)
        self.log.debug('Importance weights')
        self.log.debug( self.importanceWeights)
        self._resample( tStep)

        self.plotGraphWeightsDistribution( 0, 2, 1)
        
        for tStep in range(2, self.nTimeSteps-1):
            self.We = self.Wbar[tStep - 1]
            
            self.log.debug('Resampled Weights')
            for iParticle in range(0, self.nParticles):
                self.log.debug(self.We[:,:,iParticle])
            
            self._computeImportanceWeights(Measurement, tStep)
            self.log.debug('Importance weights')
            self.log.debug( self.importanceWeights)
            self._resample( tStep)

            self.plotGraphWeightsDistribution( 0, 2, tStep)
            self.log.info('Avg weightBar for time step ' + str(tStep) + ' is,')
            self.log.info(np.ndarray.mean( self.Wbar[ tStep],2))

        
        self.log.info('Done particle filtering! Here it goes.. The average weight particle is,')
        self.log.info(np.ndarray.mean( self.Wbar[ self.nTimeSteps - 2],2))
    """
    Class Initialization
    """
    def __init__(self):
        self.log=logging.getLogger("Flow_Net")
