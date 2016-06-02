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
The estimate class computes parameters for a universe
from a measurement, or set of measurements
"""

class GibbsEstimator:
    """
    We Histogram plots
    """
    def plotWeightDistribution(self, i, j, iGstep):
        if i == 0 and j ==2:
            vk = self.thetas[( i, j)]
            pk = self.paramPdfs[(i,j)]
            pk = pk/np.sum(pk)
            
            plt.plot( vk, pk)
            plt.axvline(x=1)
            plt.title("Weights Distribution Gibbs Step: " + str(iGstep))
            plt.xlabel("Value")
            plt.ylabel("Probability")
            
            filename = "out/gibbs/" + str(iGstep) + "_WeDistr.pdf"
            plt.savefig(filename,transparent=True) # save as pdf for higher quality
            plt.close()

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
                    self.plotWeightDistribution( iWeight, jWeight, iGStep)
                    self.We[iWeight, jWeight] = self._drawWeightSample( iWeight, jWeight)

    """
    Estimate Parameters
    """
    def estimateParameters(self, Universe, Measurement, Wl, Wu):
        # Parameter steps
        nSteps = self.nSteps
        
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
    def __init__(self, nParameterSteps, nGibbsSteps):
        self.nSteps = nParameterSteps
        self.gibbsSteps = nGibbsSteps
        self.log=logging.getLogger("Flow_Net")
