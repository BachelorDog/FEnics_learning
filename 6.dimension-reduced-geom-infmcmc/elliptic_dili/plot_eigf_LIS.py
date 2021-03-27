"""
Plot eigen-basis of global LIS subspace in Elliptic inverse problem.
Shiwei Lan @ CalTech, 2018
"""

import os,pickle
import dolfin as df
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt

from Elliptic_dili import Elliptic
import sys
sys.path.append( "../" )
from util import matplot4dolfin
matplot=matplot4dolfin()

# define the inverse problem
np.random.seed(2017)
SNR=100
elliptic = Elliptic(nx=40,ny=40,SNR=SNR)

# algorithms
algs=('DILI','aDRinfmMALA','aDRinfmHMC')
alg_names=('DILI','aDR-$\infty$-mMALA','aDR-$\infty$-mHMC')
num_algs=len(algs)
# preparation for estimates
# folder = './analysis_f_SNR'+str(SNR)
folder = './analysis_long'

# plot
num_rows=num_algs
num_dims=6
fig,axes = plt.subplots(nrows=num_rows,ncols=num_dims,sharex=True,sharey=True,figsize=(20,8))
mp.rcParams['image.cmap'] = 'jet'

for i in xrange(num_algs):
    print('Working on '+alg_names[i]+' algorithm...')
    
    # read global LIS
    fnames=[f for f in os.listdir(folder) if f.endswith('.pckl')]
    found=False
    for f_i in fnames:
        if '_'+algs[i]+'_' in f_i:
            try:
                f=open(os.path.join(folder,f_i),'rb')
                try:
                    f_read=pickle.load(f)
                except:
                    f_read=pickle.load(f,encoding='bytes')
                gLIS_eigv,gLIS_eigf=f_read[-2:]
                f.close()
                found=True
            except:
                print('pickle file broken! global LIS not read!')
    if found:
        # convert eigen-basis to function
        eig_f=df.Function(elliptic.pde.V,name="parameter")
        for j in xrange(num_dims):
            eig_f.vector()[:]=gLIS_eigf[:,j]
            plt.axes(axes.flat[i*num_dims+j])
            sub_fig=matplot.plot(eig_f)
    
# set color bar
cax,kw = mp.colorbar.make_axes([ax for ax in axes.flat])
plt.colorbar(sub_fig, cax=cax, **kw)

# save plot
# fig.tight_layout()
plt.savefig(folder+'/peigfs.png',bbox_inches='tight')
# plt.show()
