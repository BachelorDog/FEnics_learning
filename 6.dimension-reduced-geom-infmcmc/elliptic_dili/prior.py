#!/usr/bin/env python
"""
Class definition of Gaussian prior measure N(mu,_C) with mean function mu and covariance operator _C
Under conditions, each draw u(x) ~ N(mu,_C) admits a Karhunen-Loeve expansion:
u(x) = u_0 + sum_{i=1}^{infty} u_i phi_i(x) with u_i ~ N(0,lambda_i),
or u(x) = u_0 + sum_{i=1}^{infty} v_i lambda_i^(1/2) phi_i(x) with v_i ~ N(0,1);
lambda_i, phi_i(x), the eigen-pairs of the operator _C, can be defined:
by Fredholm integral operator with some kernel function, K:f->int(kf)dx,
    e.g. exponential function (kf) k(x,x'):=sigma^2exp(-||x-x'||/(2s)).
---------------------------------------------------------------
written in FEniCS 2016.2.0-dev, with backward support for 1.6.0
Shiwei Lan @ U of Warwick, 2016; @ Caltech, Sept. 2016
-------------------------------
Created July 27, 2016
-------------------------------
referred to hippylib: https://hippylib.github.io/
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2016, The EQUIP/EQUiPS projects"
__credits__ = "Umberto Villa"
__license__ = "GPL"
__version__ = "1.2"
__maintainer__ = "Shiwei Lan"
__email__ = "S.Lan@warwick.ac.uk; slan@caltech.edu; lanzithinking@outlook.com"

import os
import dolfin as df
import numpy as np
import scipy as sp
import scipy.sparse as sps

# self defined modules
import sys
sys.path.append( "../" )
from util.dolfin_gadget import get_dof_coords,vec2fun
from util.sparse_geeklet import *
from util.Eigen import *

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')
df.set_log_level(df.ERROR)

def _get_sqrtm(A,m_name='K',output_petsc=True,SAVE=True):
    """
    Get the root of matrix A.
    """
    if output_petsc and df.has_petsc4py():
        from petsc4py import PETSc
        rtA_f=os.path.join(os.getcwd(),'rt'+m_name+'_petsc_dim'+str(A.size(0))+'.dat')
        try:
            viewer = PETSc.Viewer().createBinary(rtA_f, 'r')
            rtA_petsc=df.PETScMatrix(PETSc.Mat().load(viewer))
            print('Read the root of '+{'K':'kernel','C':'covariance'}[m_name]+' successfully!')
        except:
            import scipy.linalg as spla
            rtA_sps = sps.csr_matrix(spla.sqrtm(A.array()).real)
            csr_trim0(rtA_sps,1e-10)
            rtA_petsc = df.PETScMatrix(csr2petscmat(rtA_sps))
            if SAVE:
                viewer = PETSc.Viewer().createBinary(rtA_f, 'w')
                viewer(df.as_backend_type(rtA_petsc).mat())
        return rtA_petsc
    else:
        import cPickle
        rtA_f=os.path.join(os.getcwd(),'rt'+m_name+'_sps_dim'+str(A.shape[0])+'.dat')
        try:
            f = open(rtA_f, 'rb')
            rtA_sps = cPickle.load(f)
            f.close()
            print('Read the root of '+{'K':'kernel','C':'covariance'}[m_name]+' successfully!')
        except:
            import scipy.linalg as spla
            rtA_sps = sps.csr_matrix(spla.sqrtm(A.toarray()).real)
            csr_trim0(rtA_sps,1e-10)
            if SAVE:
                f = open(rtA_f, 'wb')
                cPickle.dump(rtA_sps,f,-1)
                f.close()
        return rtA_sps

class Gaussian_prior:
    """
    Gaussian prior measure N(mu,_C) defined on 2d domain V, 
    with _C defined by the exponential kernel function K.
    """
    def __init__(self,V,sigma=1.25,s=0.0625,mean=None,rel_tol=1e-10,max_iter=100,**kwargs):
        self.V=V
        self.dim=V.dim()
        self.dof_coords=get_dof_coords(V)
        self.sigma=sigma
        self.s=s
        self.mean=mean
#         self.mpi_comm=kwargs['mpi_comm'] if 'mpi_comm' in kwargs else df.mpi_comm_world()
        self.mpi_comm=kwargs.pop('mpi_comm',df.mpi_comm_world())
        
        # mass matrix and its inverse
        M_form=df.inner(df.TrialFunction(V),df.TestFunction(V))*df.dx
        self.M=df.PETScMatrix()
        df.assemble(M_form,tensor=self.M)
        self.Msolver = df.PETScKrylovSolver("cg", "jacobi")
        self.Msolver.set_operator(self.M)
        self.Msolver.parameters["maximum_iterations"] = max_iter
        self.Msolver.parameters["relative_tolerance"] = rel_tol
        self.Msolver.parameters["error_on_nonconvergence"] = True
        self.Msolver.parameters["nonzero_initial_guess"] = False
        # square root of mass matrix
        self.rtM=self._get_rtmass()
        
        # kernel matrix and its square root
        self.K=self._get_ker()
        self.rtK = _get_sqrtm(self.K,'K')
        
        # set solvers
        for op in ['K']:
            operator=getattr(self, op)
            solver=self._set_solver(operator,op)
            setattr(self, op+'solver', solver)
        
        if mean is None:
            self.mean=df.Vector()
            self.init_vector(self.mean,0)
    
    def _get_rtmass(self,output_petsc=True):
        """
        Get the square root of assembled mass matrix M using lumping.
        --credit to: Umberto Villa
        """
        test = df.TestFunction(self.V)
        V_deg = self.V.ufl_element().degree()
        try:
            V_q_fe = df.FiniteElement('Quadrature',self.V.mesh().ufl_cell(),2*V_deg,quad_scheme='default')
            V_q = df.FunctionSpace(self.V.mesh(),V_q_fe)
        except:
            print('Use FiniteElement in specifying FunctionSpace after version 1.6.0!')
            V_q = df.FunctionSpace(self.V.mesh(), 'Quadrature', 2*self.V._FunctionSpace___degree)
        trial_q = df.TrialFunction(V_q)
        test_q = df.TestFunction(V_q)
        M_q = df.PETScMatrix()
        df.assemble(trial_q*test_q*df.dx,tensor=M_q,form_compiler_parameters={'quadrature_degree': 2*V_deg})
        ones = df.interpolate(df.Constant(1.), V_q).vector()
        dM_q = M_q*ones
        M_q.zero()
        dM_q_2fill = ones.array() / np.sqrt(dM_q.array() )
        dM_q.set_local( dM_q_2fill )
        M_q.set_diagonal(dM_q)
        mixedM = df.PETScMatrix()
        df.assemble(trial_q*test*df.dx,tensor=mixedM,form_compiler_parameters={'quadrature_degree': 2*V_deg})
        if output_petsc and df.has_petsc4py():
            rtM = df.PETScMatrix(df.as_backend_type(mixedM).mat().matMult(df.as_backend_type(M_q).mat()))
        else:
            rtM = sps.csr_matrix(mixedM.array()*dM_q_2fill)
            csr_trim0(rtM,1e-12)
        return rtM
    
    def _get_ker(self,output_petsc=True):
        """
        Get the kernel matrix K with K_ij = k(x_i,x_j).
        """
        load_success=False
        if output_petsc and df.has_petsc4py():
            from petsc4py import PETSc
            K_f=os.path.join(os.getcwd(),'K_petsc_dim'+str(self.dim)+'.dat')
            try:
                viewer = PETSc.Viewer().createBinary(K_f, 'r')
                K=df.PETScMatrix(PETSc.Mat().load(viewer))
                load_success=True
            except:
                pass
        else:
            import cPickle
            K_f=os.path.join(os.getcwd(),'K_sps_dim'+str(self.dim)+'.dat')
            try:
                f = open(K_f, 'rb')
                K = cPickle.load(f)
                f.close()
                load_success=True
            except:
                pass
        if load_success:
            print('Read the kernel successfully!')
            return K
        else:
            import scipy.spatial.distance
            K_sps = sps.csr_matrix(self.sigma**2*np.exp(-sp.spatial.distance.squareform(sp.spatial.distance.pdist(self.dof_coords))/(2*self.s)))
            csr_trim0(K_sps,1e-10)
            if output_petsc and df.has_petsc4py():
                K_petsc = df.PETScMatrix(csr2petscmat(K_sps))
                viewer = PETSc.Viewer().createBinary(K_f, 'w')
                viewer(df.as_backend_type(K_petsc).mat())
                return K_petsc
            else:
                f = open(K_f, 'wb')
                cPickle.dump(K_sps,f,-1)
                f.close()
                return K_sps
    
    def _get_cov(self,output_petsc=True):
        """
        Get the covariance matrix C with C = K M.
        """
        if output_petsc and df.has_petsc4py():
            from petsc4py import PETSc
            C_f=os.path.join(os.getcwd(),'C_petsc_dim'+str(self.dim)+'.dat')
            try:
                viewer = PETSc.Viewer().createBinary(C_f, 'r')
                C_petsc=df.PETScMatrix(PETSc.Mat().load(viewer))
                print('Read the covariance successfully!')
            except:
                C_petsc= df.PETScMatrix(df.as_backend_type(self.K).mat().matMult(df.as_backend_type(self.M).mat()))
                viewer = PETSc.Viewer().createBinary(C_f, 'w')
                viewer(df.as_backend_type(C_petsc).mat())
            return C_petsc
        else:
            import cPickle
            C_f=os.path.join(os.getcwd(),'C_sps_dim'+str(self.dim)+'.dat')
            try:
                f = open(C_f, 'rb')
                C_sps = cPickle.load(f)
                f.close()
                print('Read the covariance successfully!')
            except:
                C_sps=sps.csr_matrix(self.K.dot(self.M.array()))
                csr_trim0(C_sps)
                f = open(C_f, 'wb')
                cPickle.dump(C_sps,f,-1)
                f.close()
            return C_sps
    
    def _set_solver(self,operator,op_name=None):
        """
        Set the solver of an operator
        """
        if type(operator) is df.PETScMatrix:
            if df.dolfin_version()<='1.6.0':
                solver = df.PETScLUSolver('mumps' if df.has_lu_solver_method('mumps') else 'default')
            else:
                solver = df.PETScLUSolver(self.mpi_comm,'mumps' if df.has_lu_solver_method('mumps') else 'default')
            solver.set_operator(operator)
            solver.parameters['reuse_factorization']=True
            if op_name == 'K':
                solver.parameters['symmetric']=True
        else:
            import scipy.sparse.linalg as spsla
            solver = spsla.splu(operator.tocsc(copy=True))
        return solver
    
    def gen_vector(self,v=None):
        """
        Generate/initialize a dolfin generic vector to be compatible with the size of dof.
        """
        if v is None:
            vec = df.Vector(self.mpi_comm,self.dim)
            vec.zero()
        else:
            if type(v) is df.Vector:
                vec = df.Vector(v)
            elif type(v) is np.ndarray:
                vec = df.Vector(self.mpi_comm,len(v))
                vec[:]=np.array(v)
            else:
                df.warning('Unknown type.')
                vec=None
        return vec
    
    def init_vector(self,x,dim=0):
        """
        Initialize a vector x to be compatible with the range/domain of M.
        """
        if dim == "noise":
            self.rtM.init_vector(x, 1)
        else:
            self.M.init_vector(x,dim)
        
    def sample(self,whiten=False,add_mean=False):
        """
        Sample a random function u ~ N(0,_C)
        vector u ~ N(0,K): C=VDV^(-1), u=V sqrt(D) z ~ N(0, VDV'=CM^(-1)=K)
        """
        # whiten if asked
        if whiten:
            noise_sz=self.rtM.size(1) if type(self.rtM) is df.PETScMatrix else self.rtM.shape[1]
            noise=self.gen_vector(np.random.randn(noise_sz))
            v_vec=self.gen_vector(self.rtM*noise)
            u_vec=self.gen_vector()
            self.Msolver.solve(u_vec,v_vec)
        else:
            noise=self.gen_vector(np.random.randn(self.dim))
            u_vec=self.gen_vector(self.rtK*noise)
        # add mean if asked
        if add_mean:
            u_vec.axpy(1.,self.u2v(self.mean) if whiten else self.mean)
        
        return u_vec
    
    def logpdf(self,u,whiten=False,grad=False):
        """
        Calculate the logarithm of prior density (and its gradient) of function u.
        -.5* (u(x), _C^(-1) u(x)) = -.5* u' M C^(-1)u = -.5 u' K^(-1) u
        """
        u_m=self.gen_vector(u)
        u_m.axpy(-1.0,self.u2v(self.mean) if whiten else self.mean)
        
        Pu_m=df.Vector()
        self.M.init_vector(Pu_m,1-whiten)
        if whiten:
            self.M.mult(u_m,Pu_m)
        else:
            Pu_m=self.C_act(u_m,-1,op='K')
        
        logpri=-0.5*Pu_m.inner(u_m)
        if grad:
            gradpri=-Pu_m
            return logpri,gradpri
        else:
            return logpri
        
    def C_act(self,u_actedon,comp=1,op='K',transp=False):
        """
        Calculate operation of C^comp on vector a: a --> C^comp * a
        """
        if type(u_actedon) is np.ndarray:
            assert u_actedon.size == self.dim, "Must act on a vector of size consistent with mesh!"
            u_actedon = self.gen_vector(u_actedon)
          
        if comp==0:
            return u_actedon
        else:
            Ca=self.gen_vector()
            if comp in [1,0.5]:
                op_name={1:op,0.5:'rt'+op}[comp]
                if op_name == 'rtC' and not hasattr(self, 'rtC'):
                    self.rtC=_get_sqrtm(self._get_cov(),'C')
                multiplier=getattr(self, op_name)
                if type(multiplier) is df.PETScMatrix:
                    if transp:
                        multiplier.transpmult(u_actedon,Ca)
                    else:
                        multiplier.mult(u_actedon,Ca)
                else:
                    Ca[:]=multiplier.T.dot(u_actedon) if transp else multiplier.dot(u_actedon)
            elif comp in [-1,-0.5]:
                op_name={-1:op,-0.5:'rt'+op}[comp]+'solver'
                if op_name == 'rtCsolver' and not hasattr(self, 'rtCsolver'):
                    self.rtC=_get_sqrtm(self._get_cov(),'C')
                    self.rtCsolver=self._set_solver(self.rtC,'rtC')
                if op_name == 'rtKsolver' and not hasattr(self, 'rtKsolver'):
                    self.rtKsolver=self._set_solver(self.rtK,'rtK')
                solver=getattr(self, op_name)
                if type(solver) is df.PETScLUSolver:
                    if transp:
                        solver.solve_transpose(Ca,u_actedon)
                    else:
                        solver.solve(Ca,u_actedon)
                else:
                    Ca[:]=solver.solve(u_actedon.array(),trans='T' if transp else 'N')
            else:
                warnings.warn('Action not defined!')
                pass
            return Ca
    
    def u2v(self,u,u_ref=None):
        """
        Transform the original parameter u to the whitened parameter v:=_C^(-1/2)(u-u_ref)
        vector v:=C^(-1/2) (u-u_ref) ~ N(0, M^(-1))
        """
        b=self.gen_vector(u)
        if u_ref is not None:
            b.axpy(-1.0,u_ref)
        v=self.gen_vector()
        if not hasattr(self, 'rtCsolver'):
            self.rtC=_get_sqrtm(self._get_cov(),'C')
            self.rtCsolver=self._set_solver(self.rtC,'rtC')
        self.rtCsolver.solve(v,b)
        
        return v
    
    def v2u(self,v,u_ref=None):
        """
        Transform the whitened parameter v:=_C^(-1/2)(u-u_ref) back to the original parameter u
        """
        u=self.gen_vector()
        if not hasattr(self, 'rtC'):
            self.rtC=_get_sqrtm(self._get_cov(),'C')
        self.rtC.mult(v,u)
        if u_ref is not None:
            u.axpy(1.0,u_ref)
        
        return u
        
if __name__ == '__main__':
    from pde import *
    np.random.seed(2017)
    PDE = ellipticPDE()
    mpi_comm = PDE.mesh.mpi_comm()
    prior = Gaussian_prior(V=PDE.V,mpi_comm=mpi_comm)
    whiten = False
    u=prior.sample(whiten=whiten)
    logpri,gradpri=prior.logpdf(u, whiten=whiten, grad=True)
    print('The logarithm of prior density at u is %0.4f, and the L2 norm of its gradient is %0.4f' %(logpri,gradpri.norm('l2')))
    df.plot(vec2fun(u,PDE.V))
#     df.interactive()
    v=prior.u2v(u)
    logpri_wt,gradpri_wt=prior.logpdf(v, whiten=True, grad=True)
    print('The logarithm of prior density at whitened v is %0.4f, and the L2 norm of its gradient is %0.4f' %(logpri_wt,gradpri_wt.norm('l2')))
    df.plot(vec2fun(v,PDE.V))
#     df.interactive()
    
    whiten = True
    v=prior.sample(whiten=whiten)
    logpri,gradpri=prior.logpdf(v, whiten=whiten, grad=True)
    print('The logarithm of prior density at whitened v is %0.4f, and the L2 norm of its gradient is %0.4f' %(logpri,gradpri.norm('l2')))
    df.plot(vec2fun(v,PDE.V))
#     df.interactive()
    u=prior.v2u(v)
    logpri_wt,gradpri_wt=prior.logpdf(u, whiten=False, grad=True)
    print('The logarithm of prior density at u is %0.4f, and the L2 norm of its gradient is %0.4f' %(logpri_wt,gradpri_wt.norm('l2')))
    df.plot(vec2fun(u,PDE.V))
    df.interactive()