#!/usr/bin/env python

"""
Preliminary tests for network flow parameter estimations
"""

__author__ = "Steven Munn"
__email__ = "sjmunn@umail.ucsb.edu"
__date__ = "05-23-2016"

# -------------------- Logging Tools
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

# -------------------- Scientific Computing
from scipy.integrate import ode
from scipy.stats import norm
import numpy as np

# -------------------- Plotting and display libraries
import matplotlib as mpl
# matplotlib.use('Agg')
# Use Agg with MacOSX because of backend bug
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# -------------------- Local Modules
import Objs/Universe

# -------------------- Begin simulations

if __name__ == '__main__':
    log.info("Main script for 02Drv_GraphStruct.py has begun")

    # parameters
    trueThetas = np.array([0.75, 2])
    W = np.array([[1, 0, trueThetas[0]],
                  [0, 0, trueThetas[0]],
                  [0, trueThetas[1], 0]])
    t = np.arrange(9)    
    
    R3HwOne = Universe.Universe()
    R3HwOne.initFromMatrix(W)
    R3HwOne.simulateUniverse(t)
