"""
A wrapper of get_geom to feed Geometric quantities including nll,g,FI,cholG in MCMC samplers
Shiwei Lan @ U of Warwick, 2016
"""

import numpy as np

def geom(theta,unknown,PDE,obj,geom_opt=[0],trct_opt=-1,trct_idx=[]):
    nll=None; g=None; FI=None; cholG=None;

    # obtain gradient of data-misfit
    unknown.theta=theta
    nll,dnll,_,FI=PDE.get_geom(unknown,obj,geom_opt,trct_opt,trct_idx)

    # dimension
    D=len(theta);
    if not any(trct_idx):
        trct_idx = range(D)

    if any(s>0 for s in geom_opt):
        g=np.zeros(D)
        if 2 in geom_opt:
            g[trct_idx]=FI.dot(theta[trct_idx])
            cholG=np.linalg.cholesky(FI+np.eye(FI.shape[0]))
        if trct_opt==1: # dnll of size num_trct
            g[trct_idx]-=dnll;
        else: # dnll of size D
            g-=dnll

    return nll,g,FI,cholG
#     return {0:nll,1:g,2:(FI,cholG)}
