#!/usr/bin/env python
"""
Class definition of approximate Gaussian posterior measure N(mu,C) with mean function mu and covariance operator C
where C^(-1) = C_0^(-1) + H(u), with H(u) being Hessian (or its Gaussian-Newton approximation) of misfit;
      and the prior N(m_0, C_0)
------------------------------------------------------------
written in FEniCS 2016.2.0-dev, with backward support for 1.6.0
Shiwei Lan @ Caltech, 2016
-------------------------------
Created October 11, 2016
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2016, The EQUiPS project"
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@caltech.edu; lanzithinking@outlook.com"

# import modules
import dolfin as df
import numpy as np

from prior import *

class _lrHess:
    """
    Class of actions defined by low rank approximation of misfit Hessian (based on Gauss-Newton Hessian, not including prior precision).
    All operations are based on the (generalized) partial eigen-decomposition (H,C_0^(-1)), i.e. H = C_0^(-1) U D U^(-1); U' C_0^(-1) U = I
    """
    def __init__(self,prior,eigs):
        self.prior=prior
        self.eigs=eigs
        
    def mult(self,x,y):
        invCx=self.prior.C_act(x,-1,op='K')
        UinvCx=self.eigs[1].T.dot(invCx)
        Hx=self.prior.C_act(self.eigs[1].dot(self.eigs[0]*UinvCx),-1,op='K')
        y.zero()
        y.axpy(1.,Hx)
        return invCx
    
    def inner(self,x,y):
        Hx=self.prior.gen_vector()
        self.mult(x, Hx)
        return Hx.inner(y)
    
    def norm2(self,x):
        invCx=self.prior.C_act(x,-1,op='K')
        UinvCx=self.eigs[1].T.dot(invCx)
        return np.sum(self.eigs[0]*UinvCx**2)
    
    def solve(self,x,y):
        dum=self.eigs[1].dot(self.eigs[1].T.dot(y)/self.eigs[0])
        x.zero()
        x.axpy(1.,self.prior.gen_vector(dum))

class Gaussian_posterior_LRapp:
    """
    Low-rank Gaussian approximation of the posterior.
    """
    def __init__(self,prior,eigs,mean=None):
        self.prior=prior
        self.V=prior.V
        self.dim=prior.dim
        self.eigs=eigs # partial (generalized) eigen-decomposition of misfit Hessian H(u)
        self.Hlr=_lrHess(prior,eigs)
        self.mean=mean
        
    def gen_vector(self,v=None):
        """
        Generate/initialize a dolfin generic vector to be compatible with the size of dof.
        """
        return self.prior.gen_vector(v)
        
    def postC_act(self,u_actedon,comp=1):
        """
        Calculate the operation of (approximate) posterior covariance C^comp on vector a: a --> C^comp * a
        C^(-1) = C_0^(-1) + H = C_0^(-1) + C_0^(-1) U D U' C_0^(-1) ~ C_0^(-1) U (I+ D) U' C_0^(-1)
        C = [C_0^(-1) + H]^(-1) = C_0 - U (D^(-1) + I)^(-1) U' ~ U (I + D)^(-1) U'
        """
        if type(u_actedon) is np.ndarray:
            assert u_actedon.size == self.dim, "Must act on a vector of size consistent with mesh!"
            u_actedon = self.gen_vector(u_actedon)
        
        if comp==0:
            return u_actedon
        else:
            pCa=self.gen_vector()
            d,U=self.eigs
            if comp == -1:
                Ha=self.prior.gen_vector()
                self.Hlr.eigs=self.eigs # update eigs in low-rank Hessian approximation
                invCa=self.Hlr.mult(u_actedon,Ha)
                pCa.axpy(1.,invCa)
                pCa.axpy(1.,Ha)
            elif comp == 1:
                Ca=self.prior.C_act(u_actedon,1,op='K')
                pCa.axpy(1.,Ca)
                dum=self.gen_vector((U*(d/(d+1))).dot(U.T.dot(u_actedon)))
                pCa.axpy(-1.,dum)
            else:
                warnings.warn('Action not defined!')
                pass
            return pCa
        
    def sample(self,add_mean=False):
        """
        Sample a random function u ~ N(m,C)
        u = U (I + D)^(-1/2) z, z ~ N(0,I)
        """
        d,U=self.eigs
        noise=np.random.randn(len(d))
        
        u=self.gen_vector(U.dot(noise/np.sqrt(1+d)))
        # add mean if asked
        if add_mean:
            u.axpy(1.,self.mean)
        
        return u
        
if __name__ == '__main__':
    np.random.seed(2016)
    from Elliptic_dili import Elliptic
    # define the inverse problem
    elliptic=Elliptic(nx=40,ny=40,SNR=10)
    # get MAP
    unknown=df.Function(elliptic.pde.V)
    MAP_file=os.path.join(os.getcwd(),'MAP.h5')
    if os.path.isfile(MAP_file):
        f=df.HDF5File(elliptic.pde.mpi_comm,MAP_file,"r")
        f.read(unknown,'parameter')
        f.close()
    else:
        unknown=elliptic.get_MAP(SAVE=True)
    # get eigen-decomposition of posterior Hessian at MAP
    _,_,_,eigs=elliptic.get_geom(unknown.vector(),geom_ord=[1.5],whitened=False,threshold=1e-2)
    # define approximate Gaussian posterior
    post_Ga = Gaussian_posterior_LRapp(elliptic.prior,eigs)
    # get sample from the approximate posterior
    u = post_Ga.sample()
    df.plot(vec2fun(u,elliptic.pde.V))
    df.interactive()
    