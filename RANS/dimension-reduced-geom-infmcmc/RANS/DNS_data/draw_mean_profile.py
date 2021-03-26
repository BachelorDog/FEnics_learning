#!/bin/py

import numpy as np
import scipy
# import h5py as h5
import math
import glob
import pylab 
import sys
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
from numpy import matrix
from numpy import linalg


rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
#rcParams['xtick.major.size']=20
#rcParams['ytick.major.size']=20
#rcParams['xtick.minor.size']=10
#rcParams['ytick.minor.size']=10
#rcParams['xtick.labelsize']=30
#rcParams['ytick.labelsize']=30







x, U_cl, B_half = np.loadtxt('./PlaneJet_Re10000_axial.dat',skiprows=1, unpack=True)

plt.plot(x,U_cl)
plt.xlabel(r'$x/D$')
plt.ylabel(r'$U_{cl}/U_0$')
plt.grid(True)
plt.savefig('U_cl.pdf')
plt.close()

plt.plot(x,B_half)
plt.xlabel(r'$x/D$')
plt.ylabel(r'$y_{1/2}/D$')
plt.grid(True)
plt.savefig('y_half.pdf')
plt.close()

