#!/usr/bin/env python

"""
Preliminary tests for network flow parameter estimations
"""

__author__ = "Steven Munn"
__email__ = "sjmunn@umail.ucsb.edu"
__date__ = "05-23-2016"

# Standard
import logging

# Setup reporting
log=logging.getLogger("Flow_Net")
log.setLevel(logging.DEBUG)
# Send log output to terminal
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s : %(message)s')
console.setFormatter(formatter)
log.addHandler(console)

# Scientific Computing
from scipy.integrate import ode
from scipy.stats import norm
import numpy as np

# Plotting and display libraries
import matplotlib as mpl
# matplotlib.use('Agg') # Use Agg b/c MacOSX backend bug
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# -------------------- Begin simulations

def HwOneR3(t, M, theta1, theta2):
    dM = np.zeros((3,1))
    dM[0] = 1
    dM[1] = theta2*M[2]
    dM[2] = theta1*M[0]*M[1]
    return dM

# The ``driver`` that will integrate the ODE(s):
if __name__ == '__main__':
    t = np.arange(9)
    M = np.zeros((3,9))
    M[:,0] = np.ones(3)
    M2hat = np.zeros(9)

    M0 = np.ones((3,1))

    # parameters
    trueThetas = np.array([-0.9, 2])
    
    r = ode( HwOneR3).set_integrator('dopri5')
    r.set_initial_value([M0[0], M0[1], M0[2]], t[0])
    r.set_f_params(trueThetas[0], trueThetas[1])
    
    # Run integration
    k = 1
    while r.successful() and k < 9:
        r.integrate(r.t + 1)
 
        # Store the results to plot later
        M[0,k] = r.y[0]
        M[1,k] = r.y[1]
        M[2,k] = r.y[2]
        k+=1

    # Add noise of variance vv
    vv = 1
    M2hat = M[2,:] + np.random.rand(1,9)*np.sqrt(vv)
    M2hat = M2hat.ravel()

    nsteps = int(5000+1)
    stepSize = 2 / float(nsteps - 1)
    thetaOne = np.arange(-2-stepSize,0,stepSize)
    # 2 + stepSize to ensure it goes all the way to 2

    prior = np.ones((nsteps,1))
    posterior_t = np.zeros((8,nsteps))
    
    for j in range(0, nsteps):

	trueThetas[0] = thetaOne[j]
	trueThetas[1] = 2.5
        
        Mtest = np.zeros((3,9))
        Mtest[:,0] = np.ones(3)

        # Run integration
        rTest = ode( HwOneR3).set_integrator('dopri5')
        rTest.set_initial_value([M0[0], M0[1], M0[2]], t[0])
        rTest.set_f_params(trueThetas[0], trueThetas[1])

        k = 1
        while rTest.successful() and k < 9:
            rTest.integrate(rTest.t + 1)
            
            # Store the results to plot later
            Mtest[0,k] = rTest.y[0]
            Mtest[1,k] = rTest.y[1]
            Mtest[2,k] = rTest.y[2]
            k+=1

        posterior_t[0,j] = prior[j] * norm.pdf(M2hat[1],Mtest[2,1],vv)            
	for tStep in range(1,7):
	    posterior_t[tStep,j]=posterior_t[tStep-1,j]*norm.pdf(M2hat[tStep+1],Mtest[2,tStep+1],vv)

    for tStep in range(0,8):
        if (np.sum(posterior_t[tStep,:])/float(nsteps)) != 0:
            posterior_t[tStep,:] = posterior_t[tStep,:] / (np.sum(posterior_t[tStep,:])/float(nsteps))
        else:
            posterior_t[tStep,:] = posterior_t[tStep - 1,:]

        plt.plot(thetaOne,posterior_t[tStep,:])
        plt.ylabel('AU non-normalized probability')

        maxIdx = np.argmax(posterior_t[tStep,:])
        paramMax = thetaOne[maxIdx]

        tempText = 'The most probable parameter value at time step ' + str(tStep) + ' is:'
        log.info(tempText)
        log.info(paramMax)

        # prepare for display
        filename = "out/" + str(tStep) + "_distribution.pdf"
        plt.savefig(filename,transparent=True) # save as pdf for higher quality
        plt.close()

    
