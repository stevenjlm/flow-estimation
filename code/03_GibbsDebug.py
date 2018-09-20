#!/usr/bin/env python
"""
Script is designed to report results for Gibbs sampling
"""
# ================================================== Standard Packages
import logging

LEVEL = logging.INFO

# Setup reporting
log = logging.getLogger("Flow_Net")
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

# ================================================== Local Modules
import Objs

# ================================================== Main Simulations

if __name__ == '__main__':
    log.info("Main script for 02Drv_GraphStruct.py has begun")

    # parameters
    trueThetas = np.array([1, -1.45])
    W = np.array([[1, 0, trueThetas[0]], [0, 0, trueThetas[0]],
                  [0, trueThetas[1], 0]])
    t = np.arange(9)

    # simulate universe
    R3HwOne = Objs.Universe()
    R3HwOne.initFromMatrix(W)
    R3HwOne.simulateUniverse(t)

    # simulate measurement
    Measure = Objs.Measurement()
    Measure.simulateNodeMeasurement(R3HwOne, 2, 0.1)

    # paramater estimation bounds
    WlowerBound = np.array([[1, 0, -10], [0, 0, -10], [0, -1.45, 0]])
    WupperBound = np.array([[1, 0, 10], [0, 0, 10], [0, -1.45, 0]])

    # estimate parameters
    Est = Objs.GibbsEstimator(2000, 20)
    Est.estimateParameters(R3HwOne, Measure, WlowerBound, WupperBound)
