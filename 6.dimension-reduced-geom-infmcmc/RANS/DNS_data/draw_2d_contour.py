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



uu = uu**2
vv = vv**2
ww = ww**2

k = 0.5*(uu+vv+ww)



x = x.reshape(81,1000)
y = y.reshape(81,1000)

uu = uu.reshape(81,1000)
vv = vv.reshape(81,1000)
ww = ww.reshape(81,1000)
U = U.reshape(81,1000)
V = V.reshape(81,1000)
W = W.reshape(81,1000)
k = k.reshape(81,1000)

x = x[:,0]
y = y[0,:]

plt.plot(x)
plt.xlabel('$i$')
plt.ylabel('$x/D$')
plt.savefig('x_grid.pdf')
plt.close()

plt.plot(y)
plt.xlabel('$j$')
plt.ylabel('$y/D$')
plt.savefig('y_grid.pdf')
plt.close()

cmap1=plt.get_cmap('jet')

levels1 = LinearLocator(numticks=128).tick_values(np.amin(U),np.amax(U))
plt.contourf(x,y,np.transpose(U),levels=levels1,cmap=cmap1)
plt.colorbar()
plt.xlabel(r'$x/D$')
plt.ylabel(r'$y/D$')
plt.title(r'$U/U_{0}$')
plt.savefig('U.png',dpi=200)
plt.close()

levels1 = LinearLocator(numticks=129).tick_values(-max(abs(np.amin(V)),abs(np.amax(V))),max(abs(np.amin(V)),abs(np.amax(V))))
plt.contourf(x,y,np.transpose(V),levels=levels1,cmap=cmap1)
plt.colorbar()
plt.xlabel(r'$x/D$')
plt.ylabel(r'$y/D$')
plt.title(r'$V/U_{0}$')
plt.savefig('V.png',dpi=200)
plt.close()

levels1 = LinearLocator(numticks=129).tick_values(-max(abs(np.amin(W)),abs(np.amax(W))),max(abs(np.amin(W)),abs(np.amax(W))))
plt.contourf(x,y,np.transpose(W),levels=levels1,cmap=cmap1)
plt.colorbar()
plt.xlabel(r'$x/D$')
plt.ylabel(r'$y/D$')
plt.title(r'$W/U_{0}$')
plt.savefig('W.png',dpi=200)
plt.close()

levels1 = LinearLocator(numticks=128).tick_values(0,np.amax(uu))
plt.contourf(x,y,np.transpose(uu),levels=levels1,cmap=cmap1)
plt.colorbar()
plt.xlabel(r'$x/D$')
plt.ylabel(r'$y/D$')
plt.title(r'$u^{\prime 2}/U_0^2$')
plt.savefig('uu.png',dpi=200)
plt.close()

levels1 = LinearLocator(numticks=128).tick_values(0,np.amax(vv))
plt.contourf(x,y,np.transpose(vv),levels=levels1,cmap=cmap1)
plt.colorbar()
plt.xlabel(r'$x/D$')
plt.ylabel(r'$y/D$')
plt.title(r'$v^{\prime 2}/U_0^2$')
plt.savefig('vv.png',dpi=200)
plt.close()

levels1 = LinearLocator(numticks=128).tick_values(0,np.amax(ww))
plt.contourf(x,y,np.transpose(ww),levels=levels1,cmap=cmap1)
plt.colorbar()
plt.xlabel(r'$x/D$')
plt.ylabel(r'$y/D$')
plt.title(r'$w^{\prime 2}/U_0^2$')
plt.savefig('ww.png',dpi=200)
plt.close()

levels1 = LinearLocator(numticks=128).tick_values(0,np.amax(k))
plt.contourf(x,y,np.transpose(k),levels=levels1,cmap=cmap1)
plt.colorbar()
plt.xlabel(r'$x/D$')
plt.ylabel(r'$y/D$')
plt.title(r'$k/U_0^2$')
plt.savefig('k.png',dpi=200)
plt.close()

levels1 = LinearLocator(numticks=128).tick_values(np.amin(np.log(k)),np.amax(np.log(k)))
plt.contourf(x,y,np.transpose(np.log(k)),levels=levels1,cmap=cmap1)
plt.colorbar()
plt.xlabel(r'$x/D$')
plt.ylabel(r'$y/D$')
plt.title(r'$k/U_0^2$')
plt.savefig('log_k.png',dpi=200)
plt.close()
