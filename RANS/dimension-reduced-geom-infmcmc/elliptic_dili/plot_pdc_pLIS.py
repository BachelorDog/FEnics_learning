"""
Plot pairwise density contours in the projected LIS subspace in Elliptic inverse problem.
Shiwei Lan @ CalTech, 2018
"""

import os,pickle
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Elliptic_dili import Elliptic
# import sys
# sys.path.append( "../" )

from scipy import stats
def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)

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
        # read samples
        fnames=[f for f in os.listdir(folder) if f.endswith('.h5')]
        num_samp=10000
        prj_samp=np.zeros((num_samp,len(gLIS_eigv)))
        found=False
        samp_f=df.Function(elliptic.pde.V,name="parameter")
        bad_idx=[]
        prog=np.ceil(num_samp*(.1+np.arange(0,1,.1)))
        for f_i in fnames:
            if '_'+algs[i]+'_' in f_i:
                try:
                    f=df.HDF5File(elliptic.pde.mpi_comm,os.path.join(folder,f_i),"r")
                    for s in xrange(num_samp):
                        try:
                            f.read(samp_f,'sample_{0}'.format(s))
                        except:
                            bad_idx.append(s)
                        prj_samp[s,]=gLIS_eigf.T.dot(elliptic.prior.M*samp_f.vector())
                        if s+1 in prog:
                            print('{0:.0f}% samples have been projected into LIS subspace.'.format(np.float(s+1)/num_samp*100))
                    f.close()
                    if len(bad_idx)>0:
                        print('{0:d} bad samples encountered!'.format(len(bad_idx)))
                        prj_samp=np.delete(prj_samp,bad_idx,0)
                    found=True
                except:
                    pass
        if found:
            num_samp,dim=np.shape(prj_samp)
            idx=np.floor(np.linspace(0,num_samp-1,np.min([1e4,num_samp]))).astype(int)
            col=np.arange(np.min([6,dim]),dtype=np.int)
            mat4plot=prj_samp[np.ix_(idx,col)]
            df4plot = pd.DataFrame(mat4plot,columns=[r'$\theta_{%d}$' % k for k in col])
            
            g  = sns.PairGrid(df4plot)
            g.map_upper(plt.scatter)
            g.map_lower(sns.kdeplot, cmap="Blues_d")
            g.map_diag(sns.kdeplot, lw=3, legend=False)
            g.map_lower(corrfunc)
            
            # save plot
            g.savefig(os.path.join(folder,algs[i]+'_pdc_pLIS.png'))
#             plt.show()
