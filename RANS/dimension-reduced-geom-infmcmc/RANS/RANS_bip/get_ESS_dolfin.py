"""
Analyze MCMC samples
Shiwei Lan @ U of Warwick, 2016
"""

import os
import dolfin as dl
import numpy as np

import sys
sys.path.append( "../../" )
from util.bayesianStats import effectiveSampleSize as ess
from joblib import Parallel, delayed

# def restore_each_sample(f,samp_f,s):
#     f.read(samp_f,'sample_{0}'.format(s))
#     return samp_f.vector()

def restore_sample(mpi_comm,V,dir_name,f_name,num_samp):
    f=dl.HDF5File(mpi_comm,os.path.join(dir_name,f_name),"r")
    samp_f=dl.Function(V,name="parameter")
    samp=np.zeros((num_samp,V.dim()))
    bad_idx=[]
    prog=np.ceil(num_samp*(.1+np.arange(0,1,.1)))
    for s in xrange(num_samp):
        try:
            f.read(samp_f,'sample_{0}'.format(s))
        except:
#             print('Bad sample encountered!')
            bad_idx.append(s)
        samp[s,]=samp_f.vector()
        if s+1 in prog:
            print('{0:.0f}% samples have been restored.'.format(np.float(s+1)/num_samp*100))
#     f_read=lambda s: restore_each_sample(f,samp_f,s)
#     samp=Parallel(n_jobs=4)(delayed(f_read)(i) for i in range(num_samp))
    f.close()
    if len(bad_idx)>0:
        print('{0:d} bad samples encountered!'.format(len(bad_idx)))
        samp=np.delete(samp,bad_idx,0)
    return samp

def get_ESS(samp):
    ESS=Parallel(n_jobs=4)(map(delayed(ess), np.transpose(samp)))
    return ESS

if __name__ == '__main__':
    from RANS import RANS
    import sys
    sys.path.append( "../" )
    
    # define the PDE problem
    nozz_w=1.;nx=40;ny=80
    rans=RANS(nozz_w=nozz_w,nx=nx,ny=ny)
    rans.setup(seed=2017)
    # algorithms
#     algs=('pCN','infMALA','infHMC','DRinfmMALA','DRinfmHMC','DILI','aDRinfmMALA','aDRinfmHMC')
#     alg_names=('pCN','$\infty$-MALA','$\infty$-HMC','DR-$\infty$-mMALA','DR-$\infty$-mHMC','DILI','aDR-$\infty$-mMALA','aDR-$\infty$-mHMC')
    algs=('pCN','infMALA','DRinfmMALA','DILI','aDRinfmMALA')
    alg_names=('pCN','$\infty$-MALA','DR-$\infty$-mMALA','DILI','aDR-$\infty$-mMALA')
    num_algs=len(algs)
    # preparation for estimates
    folder = './analysis-L25W8-nofreshinit-yesparacont-2000-adph'
    fnames=[f for f in os.listdir(folder) if f.endswith('.h5')]
    num_samp=2000
    
    # calculate ESS's
    ESS=np.zeros((num_algs,rans.Vh_PARAMETER.dim()))
    found=np.zeros(num_algs,dtype=bool)
    for a in range(num_algs):
        print('Working on '+alg_names[a]+' algorithm...')
        _ESS=[]
        # samples
        for f_i in fnames:
            if '_samp_'+algs[a]+'_' in f_i:
                try:
                    samp=restore_sample(rans.mpi_comm,rans.Vh_PARAMETER,folder,f_i,num_samp)
                    _ESS_i=get_ESS(samp)
                    _ESS.append(_ESS_i)
                    found[a]=True
                except:
                    pass
        if found[a]:
            ESS[a,]=np.mean(_ESS,axis=0)
            # select some dimensions for plot
            samp_fname=os.path.join(folder,algs[a]+'_selected_samples.txt')
            if not os.path.isfile(samp_fname):
                select_indices=np.ceil(num_samp*(np.linspace(0,1,6)));select_indices[-1]=num_samp-1;select_indices=np.int_(select_indices)
                select_samples=np.vstack((select_indices,samp[:,select_indices]))
                np.savetxt(samp_fname,select_samples,delimiter=',')
    
    # save the result to file
    if any(found):
        found_idx=np.where(found)[0]
        raw_ESS=np.hstack([np.array(algs)[found_idx,None],ESS[found_idx,]])
        np.savetxt(os.path.join(folder,'raw_ESS.txt'),raw_ESS,fmt="%s",delimiter=',')
#         sumry_ESS=np.hstack((np.array(algs)[found_idx,None],np.min(ESS[found_idx,],axis=1,keepdims=True),np.median(ESS[found_idx,],axis=1,keepdims=True),np.max(ESS[found_idx,],axis=1,keepdims=True)))
        sumry_ESS=np.hstack((np.array(algs)[found_idx,None],np.min(ESS[found_idx,],axis=1,keepdims=True),np.median(ESS[found_idx,],axis=1)[:,None],np.max(ESS[found_idx,],axis=1,keepdims=True)))
        np.savetxt(os.path.join(folder,'sumry_ESS.txt'),sumry_ESS,fmt="%s",delimiter=',')
    