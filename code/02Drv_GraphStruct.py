#!/usr/bin/env python

"""
Preliminary tests for network flow parameter estimations
"""

__author__ = "Steven Munn"
__email__ = "sjmunn@umail.ucsb.edu"
__date__ = "05-23-2016"

# -------------------- Logging Tools
import logging

LEVEL = logging.DEBUG

# Setup reporting
log=logging.getLogger("Flow_Net")
log.setLevel(LEVEL)
# Send log output to terminal
console = logging.StreamHandler()
console.setLevel(LEVEL)
formatter = logging.Formatter('%(levelname)s : %(message)s')
console.setFormatter(formatter)
log.addHandler(console)

# -------------------- Scientific Computing
import numpy as np

# -------------------- Plotting and display libraries
import matplotlib as mpl
# matplotlib.use('Agg')
# Use Agg with MacOSX because of backend bug
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# -------------------- Local Modules
import sys
sys.path.insert(0, './Objs/')
import Universe


# -------------------- Begin simulations

if __name__ == '__main__':
    log.info("Main script for 02Drv_GraphStruct.py has begun")

    # parameters
    trueThetas = np.array([-1, 1.45])
    W = np.array([[1, 0, trueThetas[0]],
                  [0, 0, trueThetas[0]],
                  [0, trueThetas[1], 0]])
    t = np.arange(9)
    
    R3HwOne = Universe.Universe()
    R3HwOne.initFromMatrix(W)
    R3HwOne.simulateUniverse(t)

    Measure = Universe.Measurement()
    Measure.simulateNodeMeasurement( R3HwOne, 2, 1)

    # paramater estimation
    WlowerBound = np.array([[ 1, 0, -1],
                            [ 0, 0, -1],
                            [ 0, 1.4, 0]])
    WupperBound = np.array([[ 1, 0, -1],
                            [ 0, 0, -1],
                            [ 0, 1.6, 0]])
    Est = Universe.ParticleEstimator()
    Est.estimateParameters( R3HwOne, Measure, WlowerBound, WupperBound)
