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
from dolfin.cpp.la import LinearOperator
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2016, The EQUiPS project"
__license__ = "GPL"
__version__ = "0.4"
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

class _GA_posterior_lr:
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
    
class _solver_as_LO(df.LinearOperator):
    def __init__(self,prior,solver):
        self.prior=prior
        self.solver=solver
        
        df.LinearOperator.__init__(self,self.gen_vector(),self.gen_vector())
        
    def gen_vector(self,v=None):
        return self.prior.gen_vector(v)
    
    def init_vector(self,x, dim):
        self.prior.init_vector(x,dim)
        
    def mult(self,x,y):
        solver.solve(y,x)
    
    def _as_petscmat(self):
        if df.has_petsc4py():
            from petsc4py import PETSc
#             mat = PETSc.Mat().createPython(df.as_backend_type(self.prior.M).mat().getSizes(), comm = self.prior.mpi_comm)
            mat = PETSc.Mat().createPython(self.prior.dim, comm = self.prior.mpi_comm)
            mat.setPythonContext(self)
            return df.PETScMatrix(mat)
        else:
            df.warning('Petsc4py not installed: cannot generate PETScMatrix with specified size!')
            pass

class _GA_posterior_exct(df.LinearOperator):
    """
    Local Gaussian approximation of the posterior (no low-rank approximation of Hessian).
    """
    def __init__(self,prior,metact,rtmetact,mean=None):
        self.prior=prior
        self.V=prior.V
        self.dim=prior.dim
        self.metact=metact # Metric (misfit only) action
        self.rtmetact=rtmetact # Root-Metric (misfit only) action
        self.mean=mean
        
        self.dum=self.gen_vector()
        self.init_vector(self.dum, 1)
        
        df.LinearOperator.__init__(self,self.gen_vector(),self.gen_vector())
        
    def gen_vector(self,v=None):
        """
        Generate/initialize a dolfin generic vector to be compatible with the size of dof.
        """
        return self.prior.gen_vector(v)
    
    def init_vector(self,x, dim):
        self.prior.init_vector(x,dim)
    
    def inner(self,x,y):
        Hx = df.Vector()
        self.init_vector(Hx, 0)
        self.mult(x, Hx)
        return Hx.inner(y)
        
    def mult(self, x, y):
        """
        Posterior Hessian (PriorPrecisioin+MisfitHessian) action: y=H*x
        """
        y.zero()
        y.axpy(1., self.prior.C_act(x,-1,op='K'))
        y.axpy(1., self.metact(x))
        
    def get_solver(self,**kwargs):
        """
        Posterior covariance as the inverse of posterior Hessian
        """
#         solver=df.PETScLUSolver(self.prior.mpi_comm,'mumps' if df.has_lu_solver_method('mumps') else 'default')
#         solver.set_operator(self._as_petscmat())
#         solver.parameters['reuse_factorization']=True
#         solver.parameters['symmetric']=True
        
        preconditioner=kwargs.pop('preconditioner',None)
        if preconditioner:
            solver = df.PETScKrylovSolver("default", "default")
            solver.set_operators(self,preconditioner)
        else:
            solver = df.PETScKrylovSolver("cg", "none") # very slow without proper preconditioner
            solver.set_operator(self)
        solver.parameters["maximum_iterations"] = 1000
        solver.parameters["relative_tolerance"] = 1e-7
        solver.parameters["error_on_nonconvergence"] = True
        solver.parameters["nonzero_initial_guess"] = False
        
        return solver
    
    def sample(self,add_mean=False):
        self.dum=self.prior.C_act(np.random.randn(self.dim),-0.5,op='K')
#         self.dum=self.prior.C_act(self.prior.sample(whiten=False),-1,op='K')
        self.dum.axpy(1.,self.rtmetact(self.prior.sample(whiten=True)))
#         Hsolver=self.get_solver(preconditioner=_solver_as_LO(self.prior, self.prior.Ksolver)) # not working
        Hsolver=self.get_solver()
        u=self.gen_vector()
        self.init_vector(u, 1)
        Hsolver.solve(u,self.dum)
        # add mean if asked
        if add_mean:
            u.axpy(1.,self.mean)
        return u
    
    def _as_petscmat(self):
        if df.has_petsc4py():
            from petsc4py import PETSc
            mat = PETSc.Mat().createPython(df.as_backend_type(self.prior.M).mat().getSizes(), comm = self.prior.mpi_comm)
#             mat = PETSc.Mat().createPython(self.dim, comm = self.prior.mpi_comm)
            mat.setPythonContext(self)
            return df.PETScMatrix(mat)
        else:
            df.warning('Petsc4py not installed: cannot generate PETScMatrix with specified size!')
            pass
    
def Gaussian_apx_posterior(prior,**kwargs):
    """
    Class switcher.
    """
    eigs=kwargs.pop('eigs',None)
    if eigs:# is not None:
        return _GA_posterior_lr(prior,eigs,**kwargs)
    metact=kwargs.pop('metact',None)
    rtmetact=kwargs.pop('rtmetact',None)
    if metact and rtmetact:
        return _GA_posterior_exct(prior,metact,rtmetact,**kwargs)
    else:
        df.error('Definition not specified!')
        
if __name__ == '__main__':
#     np.random.seed(2017)
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
    
    import time
    start = time.time()
    # get eigen-decomposition of posterior Hessian at MAP
    _,_,_,eigs=elliptic.get_geom(unknown.vector(),geom_ord=[1.5],whitened=False,k=100)#threshold=1e-2)
    # define approximate Gaussian posterior
    post_Ga = Gaussian_apx_posterior(elliptic.prior,eigs=eigs)
    # get sample from the approximate posterior
    u = post_Ga.sample()
    df.plot(vec2fun(u,elliptic.pde.V))
    end = time.time()
    print('Time used is %.4f' % (end-start))
    
    start = time.time()
    # get metact/rtmetact
    _,_,metact,rtmetact=elliptic.get_geom(unknown.vector(),geom_ord=[1.5],whitened=False)
    # define approximate Gaussian posterior
    post_Ga = Gaussian_apx_posterior(elliptic.prior,metact=metact,rtmetact=rtmetact)
    # get sample from the approximate posterior
    u = post_Ga.sample()
    df.plot(vec2fun(u,elliptic.pde.V))
    end = time.time()
    print('Time used is %.4f' % (end-start))
    
    df.interactive()
    