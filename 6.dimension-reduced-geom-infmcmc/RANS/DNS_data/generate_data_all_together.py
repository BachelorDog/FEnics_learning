#!/bin/py

import numpy as np
import scipy
import h5py as h5
import math
import glob
import pylab 
import sys
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import LinearLocator
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







x, y, U, uu, W, ww, V, vv = np.loadtxt('./PlaneJet_Re10000_profiles.dat',skiprows=1, unpack=True)

x, z_norm, uv, z = np.loadtxt('./PlaneJet_Re10000_uw.dat',skiprows=1,unpack=True)

x_temp, U_cl, B_half = np.loadtxt('./PlaneJet_Re10000_axial.dat',skiprows=1,unpack=True)





uu = uu**2
vv = vv**2
ww = ww**2

k = 0.5*(uu+vv+ww)

#print x[0:1001]

for i in range(81):
    dummy = 1.0*(U_cl[i]*U_cl[i])
    uv[i*1000:(i+1)*1000] = uv[i*1000:(i+1)*1000]*dummy
    
f = open('PlaneJet_data_all.dat','w')

f.write("       x        ")
f.write("       y        ")
f.write("       U        ")
f.write("       V        ")
f.write("       uu       ")
f.write("       vv       ")
f.write("       ww       ")
f.write("       uv       ")
f.write("       k        \n")
f.write("---------------------------------------------------------------------------------------------------------------------------------------------------\n")


for i in range(len(x)):
    f.write("%16.5e" % x[i])
    f.write("%16.5e" % y[i])
    f.write("%16.5e" % U[i])
    f.write("%16.5e" % V[i])
    f.write("%16.5e" % uu[i])
    f.write("%16.5e" % vv[i])
    f.write("%16.5e" % ww[i])
    f.write("%16.5e" % uv[i])
    f.write("%16.5e" % k[i])
    f.write("\n")
f.close
          

