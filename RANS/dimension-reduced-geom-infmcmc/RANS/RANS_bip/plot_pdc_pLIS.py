"""
Plot pairwise density contours in the projected LIS subspace in inverse RANS problem.
Shiwei Lan @ CalTech, 2018
"""

import os,pickle
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import sys
sys.path.append( "../../" )
import hippylib as hp
from RANS import RANS

from scipy import stats
def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)

# define the inverse problem
PARAMETER =1
# define the PDE problem
nozz_w=1.;nx=40;ny=80
rans=RANS(nozz_w=nozz_w,nx=nx,ny=ny)
rans.setup(seed=2017)

# algorithms
algs=('DILI','aDRinfmMALA','aDRinfmHMC')
alg_names=('DILI','aDR-$\infty$-mMALA','aDR-$\infty$-mHMC')
num_algs=len(algs)
# preparation for estimates
folder = './analysis-L25W8-nofreshinit-yesparacont-2000'

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
                eig_dim=f_read[-3][-1]
                f.close()
                found=True
            except:
                print('pickle file broken! LIS dimension not read!')
    fnames=[f for f in os.listdir(folder) if f.endswith('.h5')]
    found=False
#     eig_f=df.Function(rans.Vh_PARAMETER,name="eigenbasis")
#     gLIS_eigf=hp.linalg.MultiVector(eig_f.vector(),eig_dim)
    gLIS_eigf=hp.linalg.MultiVector(rans.model_stat.generate_vector(PARAMETER),eig_dim)
    for f_i in fnames:
        if 'gLIS_'+algs[i]+'_' in f_i:
            try:
                f=df.HDF5File(rans.mpi_comm,os.path.join(folder,f_i),"r")
                for s in xrange(eig_dim):
#                     f.read(eig_f.vector(),'/VisualisationVector/{0}'.format(s),False)
#                     gLIS_eigf[s][:]=eig_f.vector()
                    f.read(gLIS_eigf[s],'/VisualisationVector/{0}'.format(s),False)
                f.close()
                found=True
            except:
                print('HDF5 file broken! global LIS not read!')
    if found:
        # read samples
        fnames=[f for f in os.listdir(folder) if f.endswith('.h5')]
        num_samp=2000
        prj_samp=np.zeros((num_samp,eig_dim))
        found=False
        samp_f=df.Function(rans.Vh_PARAMETER,name="parameter")
        bad_idx=[]
        prog=np.ceil(num_samp*(.1+np.arange(0,1,.1)))
        for f_i in fnames:
            if '_samp_'+algs[i]+'_' in f_i:
                try:
                    f=df.HDF5File(rans.mpi_comm,os.path.join(folder,f_i),"r")
                    for s in xrange(num_samp):
                        try:
                            f.read(samp_f,'sample_{0}'.format(s))
                        except:
                            bad_idx.append(s)
                        prj_samp[s,]=gLIS_eigf.dot_v(rans.prior.M*samp_f.vector())
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
