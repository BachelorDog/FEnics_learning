#!/usr/bin/env python
"""
Class definition of Elliptic PDE model in the DILI paper by Cui et~al (2016)
written in FEniCS 1.7.0-dev, with backward support for 1.6.0, portable to other PDE models
Shiwei Lan @ U of Warwick, 2016
-----------------------------------
The purpose of this script is to obtain geometric quantities, misfit, its gradient and the associated metric (Gauss-Newton) using adjoint methods.
--To run demo:                     python Elliptic.py # to compare with the finite difference method
--To initialize problem:     e.g.  elliptic=Elliptic(args); coeff=elliptic.coefficient(args)
--To obtain observations:          obs,idx,loc,sd_noise=elliptic.get_obs([coeff]) # observation values, dof indices, locations, and standard deviation of noise resp.
--To define data misfit class:     misfit=elliptic.data_misfit(args)
--To obtain geometric quantities:  nll,dnll,Fv,FI = elliptic.get_geom # misfit value, gradient, metric action and metric resp.
                                   which calls soln_fwd, get_grad (soln_adj), get_metact (soln_fwd2,soln_adj2), and get_met resp.
--To save PDE solutions:           elliptic.save()
                                   fwd: forward solution; adj: adjoint solution; fwd2: 2nd order forward; adj2: 2nd order adjoint.
--To plot PDE solutions:           elliptic.plot()
-----------------------------------
Created May 18, 2016
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2016, The EQUIP/EQUiPS projects"
__license__ = "GPL"
__version__ = "5.1"
__maintainer__ = "Shiwei Lan"
__email__ = "S.Lan@warwick.ac.uk; lanzithinking@outlook.com"

# import modules
from dolfin import *
import ufl
# from dolfin_adjoint import *
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg

# self defined modules
import sys
sys.path.append( "../" )
from util import *

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')
set_log_level(ERROR)

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

class Elliptic:
    def __init__(self,nx=40,ny=40,nugg=1.0e-20):
        # 1. Define the Geometry
        # mesh size
        self.nx=nx; self.ny=ny;
        self.nugg = Constant(nugg)
        # set FEM
        self.set_FEM()

        # count PDE solving times
        self.soln_count = np.zeros(4)
        # 0-3: number of solving (forward,adjoint,2ndforward,2ndadjoint) equations respectively

    def set_FEM(self):
        self.mesh = UnitSquareMesh(self.nx, self.ny)

        # boundaries
        self.boundaries = FacetFunction("size_t", self.mesh, 0)
        self.ds = ds(subdomain_data=self.boundaries)

        # 2. Define the finite element spaces and build mixed space
        try:
            V_fe = FiniteElement("CG", self.mesh.ufl_cell(), 2)
            L_fe = FiniteElement("CG", self.mesh.ufl_cell(), 2)
            self.V = FunctionSpace(self.mesh, V_fe)
            self.W = FunctionSpace(self.mesh, V_fe * L_fe)
        except TypeError:
            print('Warning: ''MixedFunctionSpace'' has been deprecated in DOLFIN version 1.7.0.')
            print('It will be removed from version 2.0.0.')
            self.V = FunctionSpace(self.mesh, 'CG', 2)
            L = FunctionSpace(self.mesh, 'CG', 2)
            self.W = self.V * L

        # 3. Define boundary conditions
        bc_lagrange = DirichletBC(self.W.sub(1), Constant(0.0), "fabs(x[0])>2.0*DOLFIN_EPS & fabs(x[0]-1.0)>2.0*DOLFIN_EPS & fabs(x[1])>2.0*DOLFIN_EPS & fabs(x[1]-1.0)>2.0*DOLFIN_EPS")

        self.ess_bc = [bc_lagrange]

        # Create adjoint boundary conditions (homogenized forward BCs)
        def homogenize(bc):
            bc_copy = DirichletBC(bc)
            bc_copy.homogenize()
            return bc_copy
        self.adj_bcs = [homogenize(bc) for bc in self.ess_bc]
        
        # dof coordinates
        try:
            self.dof_coords = self.V.tabulate_dof_coordinates() # post v1.6.0
        except AttributeError:
            print('Warning: tabulate_all_coordinates of GenericDofMap object will be deprecated after version 1.6.0!')
            self.dof_coords = self.V.dofmap().tabulate_all_coordinates(self.mesh)
        self.dof_coords.resize((self.V.dim(), self.mesh.geometry().dim()))

    # subclass of Expression with varying parameters
    class _coefficient_fb(Expression): # Karhunen-Loeve expansion of coefficient # TODO: write cpp code
        """
        Coefficient function defined by K-L expansion with Fourier basis
        """
        def __new__(cls,*args,**kwargs):
            inst = super(Expression, cls).__new__(cls)
            if 'value_shape' in kwargs:
                inst._value_shape = (kwargs['value_shape'],)
            return inst
        def __init__(self,out_obj,theta,sigma=1.0,alpha=0.01,s=1.1,d=False,**kwargs):
            self.out_obj=out_obj
            self.theta=theta
            self.sigma=sigma
            self.alpha=alpha
            self.s=s
            self.d=d
            self.l=len(self.theta)
        # K-L expansion of theta ~ GP(0,C)
        def eval(self,value,x):
            lx=np.ceil(sqrt(self.l))
            ly=np.ceil(self.l/lx)
            seq0lx = np.arange(lx,dtype=np.float)
            seq0ly = np.arange(ly,dtype=np.float)
            eigv = self.sigma*pow(self.alpha+pi**2*(seq0lx[:,None]**2+seq0ly[None,]**2),-self.s/2)
            eigf = np.cos(pi*x[0]*seq0lx[None,:,None]) * np.cos(pi*x[1]*seq0ly[None,None,])
            dlogc = np.reshape(eigf,(-1,eigv.size))*eigv.flatten()
            c = np.exp(dlogc.dot(self.theta))
            if self.d and self._value_shape==dlogc.size:
                value[:] = dlogc.flatten()
            else:
                value[0] = c
        def value_shape(self): # called after __new__ and before __init__!
            return self._value_shape
        def get_coeff(self):
            coeff_fun = interpolate(self,self.out_obj.V)
            coeff_vec = coeff_fun.vector().array()
            dlogcoeff = None
            if self.d:
                lx=np.ceil(sqrt(self.l))
                ly=np.ceil(self.l/lx)
                seq0lx = np.arange(lx,dtype=np.float)
                seq0ly = np.arange(ly,dtype=np.float)
                eigv = self.sigma*pow(self.alpha+pi**2*(seq0lx[:,None]**2+seq0ly[None,]**2),-self.s/2)
                eigv = eigv.flatten()
                eigf = np.cos(pi*self.out_obj.dof_coords[:,0,None,None]*seq0lx[None,:,None]) * np.cos(pi*self.out_obj.dof_coords[:,1,None,None]*seq0ly[None,None,])
                eigf = np.reshape(eigf,(-1,eigv.size))
                dlogcoeff = eigf*eigv
            return coeff_fun,coeff_vec,dlogcoeff
    
    class _kernel(Expression): # exponential kernel function of Gaussian prior measure
        def __new__(cls,*args,**kwargs):
            inst = super(Expression, cls).__new__(cls)
            if 'value_shape' in kwargs:
                inst._value_shape = (kwargs['value_shape'],)
            return inst
        def __init__(self,out_obj,sigma=1.25,s=0.0625,**kwargs):
            self.out_obj=out_obj
            self.sigma=sigma
            self.s=s
            if 'degree' in kwargs:
                self.deg=kwargs['degree']
            else:
                self.deg=self.out_obj.V.ufl_element().degree()
            self.ker_x=Expression('pow(sigma,2)*exp(-sqrt(pow(x[0]-x0,2)+pow(x[1]-x1,2))/(2*s))',x0=0,x1=1,sigma=self.sigma,s=self.s,degree=self.deg)
            self.fom_x=inner(self.ker_x,TestFunction(self.out_obj.V))*dx
        def eval(self,value,x):
            self.ker_x.x0=x[0]
            self.ker_x.x1=x[1]
            value[:]=assemble(self.fom_x).array()
        def get_eigen(self,n=100):
            print("Assembling kernel function...")
            ker_mat=np.array([self(x) for x in self.out_obj.dof_coords])
            # convert the assembled kernel matrix to PETScMatrix format
            ker_sps = sps.csr_matrix(ker_mat)
            # pruning small values
            ker_sps.data *= abs(ker_sps.data)>=1e-8
            ker_sps.eliminate_zeros()
            # solve the associated eigen-problem
            if has_petsc4py() and has_slepc():
                # using SLEPc
                from petsc4py import PETSc
                csr2petscmat = lambda matrix: PETSc.Mat().createAIJ(size=matrix.shape,csr=(matrix.indptr,matrix.indices,matrix.data))
                ker_petsc = PETScMatrix(csr2petscmat(ker_sps))
                print("Computing the first %d eigenvalues..." % n)
                # Create eigen-solver
                eigen = SLEPcEigenSolver(ker_petsc)
                eigen.solve(n)
                return eigen
            else:
                warnings.warn('petsc4py or SLEPc not found! Using scipy.sparse.linalg.eigs...')
                eigenvalues,eigenvectors = sps.linalg.eigsh(ker_sps,n)
                return eigenvalues,eigenvectors
#         def __reduce__(self): # hack to return a picklable class
#             # return a class which can return this class when called with the appropriate tuple of arguments
#             state=self.__dict__.copy()
#             return (_get_nested(),(self.out_obj,self.__class__.__name__,),state,)
    
    def kernel(self,sigma=1.25,s=0.0625,**kwargs):
        return self._kernel(self,sigma,s,value_shape=self.V.dim(),**kwargs)
    
    class _coefficient_kf(object): # Karhunen-Loeve expansion of coefficient function
        """
        Coefficient function defined by K-L expansion through exponential kernel function
        """
        def __init__(self,out_obj,theta,sigma=1.25,s=0.0625,**kwargs):
            self.out_obj=out_obj
            self.theta=theta
            self.sigma=sigma
            self.s=s
            self.l=len(self.theta)
            if {'ker','eigen'}<=set(kwargs):
                self.ker=ker
                self.eigen=Eigene
            else:
                self.ker=self.out_obj.kernel(sigma=self.sigma,s=self.s,**kwargs)
                self.eigen=self.ker.get_eigen(self.l)
        def get_coeff(self):
            if type(self.eigen) is dolfin.cpp.la.SLEPcEigenSolver:
                eigv=np.zeros(self.l)
                eigf=np.zeros((self.out_obj.V.dim(),self.l))
                for k in range(self.l):
                    eigv[k],_,eigf[:,k],_=self.eigen.get_eigenpair(k)
            else:
                eigv,eigf=self.eigen
            dlogcoeff = eigf*np.sqrt(eigv)
            coeff_vec = np.exp(dlogcoeff.dot(self.theta))
            coeff_fun = Function(self.out_obj.V)
            coeff_fun.vector()[:] = coeff_vec
            return coeff_fun,coeff_vec,dlogcoeff
    
    def coefficient(self,theta,kl_opt='fb',**kwargs):
        coeff_=getattr(self, '_coefficient_'+kl_opt)
        print('Coefficient is defined by K-L expansion using '+ {'fb':'Fourier basis','kf':'exponential kernel function'}[kl_opt]+'!')
        return coeff_(self,theta,**kwargs)
    
    class source_term(Expression):
        def __init__(self,mean=np.array([[0.3,0.3], [0.7,0.3], [0.7,0.7], [0.3,0.7]]),sd=0.05,wts=[2,-3,3,-2],**kwargs):
            self.mean=mean
            self.sd=sd
            self.wts=wts
        def eval(self,value,x):
            pdfs=1/(2*pi*self.sd**2)*np.exp(-.5*np.sum((self.mean-x)**2,axis=1)/self.sd**2)
            value[0]=pdfs.dot(self.wts)

    def set_forms(self,coeff,obj=None,geom_ord=[0],trt_opt=-1,trt_idx=[]):
        if not (any(trt_idx) or 0<=trt_opt<=2):
            trt_idx = range(coeff.l)
        if any(s>=0 for s in geom_ord):
            ## forms for forward equation ##
            # 4. Define variational problem
            # functions
            if not hasattr(self, 'states_fwd'):
                self.states_fwd = Function(self.W)
            # u, l = split(self.states_fwd)
            u, l = TrialFunctions(self.W)
            v, m = TestFunctions(self.W)
            f = self.source_term(degree=2)
            # variational forms
            if 'true' in str(type(coeff)):
                u_coeff = interpolate(coeff, self.V)
            else:
                u_coeff,coeff_vec,_ = coeff.get_coeff()
            self.F = u_coeff*inner(grad(u), grad(v))*dx + (u*m + v*l)*self.ds - f*v*dx + self.nugg*l*m*dx
            # self.dFdstates = derivative(self.F, self.states_fwd) # Jacobian
#             self.a = u_coeff*inner(grad(u), grad(v))*dx + (u*m + v*l)*self.ds + self.nugg*l*m*dx
#             self.L = f*v*dx
        if any(s>=1 for s in geom_ord):
            ## forms for adjoint equation ##
            # Set up the objective functional J
#             u,_,_ = split(self.states_fwd)
#             J_form = obj.form(u)
            # Compute adjoint of forward operator
            F2 = action(self.F, self.states_fwd)
            self.dFdstates = derivative(F2, self.states_fwd)    # linearized forward operator
            args = ufl.algorithms.extract_arguments(self.dFdstates) # arguments for bookkeeping
            self.adj_dFdstates = adjoint(self.dFdstates, reordered_arguments=args) # adjoint linearized forward operator
#             self.dJdstates = derivative(J_form, self.states_fwd, TestFunction(self.W)) # derivative of functional with respect to solution
#             self.dirac_1 = obj.ptsrc(u,1) # dirac_1 cannot be initialized here because it involves evaluation
            ## forms for gradient ##
            self.dFdunknown = derivative(F2, u_coeff)
            self.adj_dFdunknown = adjoint(self.dFdunknown)

            # obtain compressed dunknown/dtheta, i.e. du_coeff/dtheta
            if not hasattr(self, 'dlogcoeff_mat'):
#                 if type(coeff) is self._coefficient_kf:
#                     _,_,self.dlogcoeff_mat = coeff.get_coeff()
#                 else:
#                     lx=np.ceil(sqrt(coeff.l))
#                     ly=np.ceil(coeff.l/lx)
#                     seq0lx = np.arange(lx,dtype=np.float)
#                     seq0ly = np.arange(ly,dtype=np.float)
#                     eigv = coeff.sigma*pow(coeff.alpha+pi**2*(seq0lx[:,None]**2+seq0ly[None,]**2),-coeff.s/2)
#                     eigv = eigv.flatten()
#                     eigf = np.cos(pi*self.dof_coords[:,0,None,None]*seq0lx[None,:,None]) * np.cos(pi*self.dof_coords[:,1,None,None]*seq0ly[None,None,])
#                     eigf = np.reshape(eigf,(-1,eigv.size))
#                     self.dlogcoeff_mat = eigf*eigv
# #                     dlogcoeff = self.coefficient(theta=coeff.theta,d=True,degree=2,value_shape=coeff.l)
# #                     dlogcoeff_f = interpolate(dlogcoeff, TensorFunctionSpace(self.mesh,'CG',self.V.ufl_element().degree(),shape=dlogcoeff.ufl_shape)) # slow
# #                     self.dlogcoeff_mat = np.reshape(dlogcoeff_f.vector(),(-1,dlogcoeff_f.ufl_shape[0]))
                
                if type(coeff) is self._coefficient_fb:
                    d_=coeff.d; coeff.d=True
                _,_,self.dlogcoeff_mat = coeff.get_coeff()
                if type(coeff) is self._coefficient_fb:
                    coeff.d=d_
                
                if trt_opt==1:# or (trt_opt==2 and any(s>1 for s in geom_ord)):
                    self.dlogcoeff_mat = self.dlogcoeff_mat[:,trt_idx]
            self.dunknowndtheta_mat = self.dlogcoeff_mat*coeff_vec[:,None]
#             dunknowndtheta_f = project(coeff*self.dlogcoeff,TensorFunctionSpace(self.mesh,'CG',self.V.ufl_element().degree(),shape=self.dlogcoeff.ufl_shape)) # very slow
#             self.dunknowndtheta_mat = np.reshape(dunknowndtheta_f.vector(),(-1,dunknowndtheta_f.ufl_shape[0]))
#             print(abs(self.dunknowndtheta_mat-dunknowndtheta_mat).max())
#             if trt_opt==1:
#                 self.dunknowndtheta_mat = self.dunknowndtheta_mat[:,trt_idx]

        if any(s>1 for s in geom_ord):
#             ## forms for 2nd adjoint equation ##
# #             self.d2Jdstates = derivative(self.dJdstates, self.states_fwd) # 2nd order derivative of functional with respect to solution
#             self.dirac_2 = obj.ptsrc(geom_ord=2) # dirac_1 cannot be initialized here because it is independent of u
            # create sparse matrix with scipy
            if hasattr(self, 'dlogcoeff_mat') and not hasattr(self, 'dlogcoeff_sps'):
                dlogcoeff2sps = self.dlogcoeff_mat
                if trt_opt==2:
                    dlogcoeff2sps = dlogcoeff2sps[:,trt_idx]
                self.dlogcoeff_sps = sps.csr_matrix(dlogcoeff2sps)
                # pruning small values
                self.dlogcoeff_sps.data *= abs(self.dlogcoeff_sps.data)>=1e-8
                self.dlogcoeff_sps.eliminate_zeros()

            # multiplication of a vector to each column of a csr_matrix
            # solution take from http://stackoverflow.com/questions/12237954/multiplying-elements-in-a-sparse-array-with-rows-in-matrix
            self.dunknowndtheta_sps = self.dlogcoeff_sps.copy()
#             self.dunknowndtheta_sps.data *= u_coeff.vector().array().repeat(np.diff(self.dunknowndtheta_sps.indptr))
            self.dunknowndtheta_sps.data *= coeff_vec.repeat(np.diff(self.dunknowndtheta_sps.indptr))

#             dunknowndtheta_sps = sps.csr_matrix(self.dunknowndtheta_mat)
#             dunknowndtheta_sps.data *= abs(dunknowndtheta_sps.data)>=1e-8
#             dunknowndtheta_sps.eliminate_zeros()

    def soln_fwd(self):
        # 5. Solve (non)linear variational problem
        # solve(self.F==0,self.states_fwd,self.ess_bc,J=self.dFdstates)
#         self.states_fwd = Function(self.W)
        solve(lhs(self.F)==rhs(self.F),self.states_fwd,self.ess_bc)
#         solve(self.a==self.L,self.states_fwd,self.ess_bc)
        self.soln_count[0] += 1
        u_fwd, l_fwd = split(self.states_fwd)
        return u_fwd, l_fwd

    # true transmissivity field
    class true_coeff(Expression):
        def __init__(self,**kwargs):
            self.truth_area1 = lambda x: .6<= x[0] <=.8 and .2<= x[1] <=.4
            self.truth_area2 = lambda x: (.8-.3)**2<= (x[0]-.8)**2+(x[1]-.2)**2 <=(.8-.2)**2 and x[0]<=.8 and x[1]>=.2
        def eval(self,value,x):
            if self.truth_area1(x) or self.truth_area2(x):
                value[0] = exp(-1)
            else:
                value[0] = 1

    def get_obs(self,unknown=None,SNR=10):
        print('Obtaining observations on a refined (double-sized) mesh...')
        if unknown is None:
            unknown = self.true_coeff(degree=0)
        # obtain the solution of u in a finer mesh
        self.nx*=2; self.ny*=2
        self.set_FEM()
        if type(unknown) is self._coefficient_kf:
            ker_=unknown.ker
            eigen_=unknown.eigen
            unknown.ker=self.kernel(sigma=unknown.sigma,s=unknown.s,degree=ker_.deg)
            unknown.eigen=unknown.ker.get_eigen(unknown.l)
        self.set_forms(unknown)
        _,_=self.soln_fwd()
        u,_=self.states_fwd.split(True)
        u_vec = u.vector()
        # choose locations based on a set of coordinates
        sq = np.arange(.2,.6+.1,.1)
        X, Y = np.meshgrid(sq, sq)
        loc = np.array([X.flatten(),Y.flatten()]).T
        idx,loc,_ = self._in_dof(loc,tol=1e-6) # dof index in V_velocity
#         print(idx)
        # obtain observations
        if idx is not None:
            sol_on_loc = u_vec[idx]
        else:
            sol_on_loc = [u(list(x)) for x in loc]
        sd_noise = u_vec.max()/SNR
        obs = sol_on_loc + sd_noise*np.random.randn(len(sol_on_loc))
        # reset to the original coarser mesh
        self.nx/=2; self.ny/=2
        self.set_FEM()
        if type(unknown) is self._coefficient_kf:
            unknown.ker=ker_
            unknown.eigen=eigen_
        self.states_fwd = interpolate(self.states_fwd,self.W)
#         del self.states_fwd
        # update indices, locations and observations
        idx,loc,rel_idx = self._in_dof(loc,tol=1e-6)
#         print(idx)
        obs = obs[rel_idx]
        print('%d observations have been obtained!' % len(idx))
        return obs,idx,loc,sd_noise

    def _in_dof(self,points,V=None,tol=2*DOLFIN_EPS): # generic function to determine whether points are nodes where dofs are defined and output those dofs
        # obtain coordinates of dofs
        if V is None:
            dof_coords=self.dof_coords
        else:
            n = V.dim() # V should NOT be mixed function space! Unless you know what you are doing...
            d = self.mesh.geometry().dim()
            if V.num_sub_spaces()>1:
                print('Warning: Multiple dofs associated with each point, unreliable outputs!')
            try:
                dof_coords = V.tabulate_dof_coordinates() # post v1.6.0
            except AttributeError:
                print('Warning: tabulate_all_coordinates of GenericDofMap object will be deprecated after version 1.6.0!')
                dof_coords = V.dofmap().tabulate_all_coordinates(self.mesh)
            dof_coords.resize((n, d))
        # check whether those points are close to nodes where dofs are defined
        pdist_pts2dofs = np.einsum('ijk->ij',(points[:,None,:]-dof_coords[None,:,:])**2)
        idx_in_dof = np.argmin(pdist_pts2dofs,axis=1)
        rel_idx_in = np.where(np.einsum('ii->i',pdist_pts2dofs[:,idx_in_dof])<tol**2)[0] # index relative to points
        idx_in_dof = idx_in_dof[rel_idx_in]
        loc_in_dof = points[rel_idx_in,]
        return idx_in_dof,loc_in_dof,rel_idx_in

    class _data_misfit(object):
        def __init__(self,out_obj,obs,prec,idx=None,loc=None):
            self.out_obj=out_obj
            self.obs = obs
            self.prec = prec
            self.idx = idx
            self.loc = loc
#             # define point (Dirac) measure centered at observation locations, but point integral is limited to CG1
#             # error when compiling: Expecting test and trial spaces to only have dofs on vertices for point integrals.
#             pts_domain = VertexFunction("size_t", self.out_obj.mesh, 0) # limited to vertices, TODO: generalize to e.g. dofs nodal points
# #             pts_nbhd = AutoSubDomain(lambda x: any([near(x[0],p[0]) and near(x[1],p[1]) for p in self.loc]))
#             pts_nbhd = AutoSubDomain(lambda x: any([Point(x).distance(Point(p))<2*DOLFIN_EPS for p in self.loc]))
#             pts_nbhd.mark(pts_domain, 1)
#             self.dpm = dP(subdomain_data=pts_domain)
            # find global dof of observations
#             idx_dirac_local,_,self.idx_dirac_rel2V = self.out_obj._in_dof(self.loc, self.out_obj.V) # idx_dirac_rel2Vv: indices relative to V
            idx_dirac_local = self.idx # indices relative to V
            sub_dofs = self.out_obj.W.sub(0).dofmap().dofs() # dof map: V --> W
            self.idx_dirac_global = sub_dofs[idx_dirac_local] # indices relative to W
        def extr_sol_vec(self,u):
            # u_vec: solution vector on observation locations
            if type(u) is ufl.indexed.Indexed:
                u_vec = [u(list(x)) for x in self.loc]
            elif type(u) is dolfin.functions.function.Function:
                if self.idx is not None:
                    u_vec = u.vector()[self.idx]
                elif self.loc is not None:
                    u_vec = [u(list(x))[0] for x in self.loc]
            elif type(u) is dolfin.cpp.la.GenericVector or np.ndarray:
                u_vec = u[self.idx]
            else:
                raise Exception('Check the type of u! Either the indeces or the locations of observations are needed!')
            return np.array(u_vec)
        def eval(self,u):
            u_vec = self.extr_sol_vec(u)
            diff = u_vec-self.obs
            val = 0.5*self.prec*diff.dot(diff)
            return val
#         def func(self,u):
#             if type(u) is not ufl.indexed.Indexed:
#                 print('Warning: use split() instead of .split(True) to get u!')
#             f_ind = Function(self.out_obj.V)
# #             f_ind.vector()[:] = 0
#             f_ind.vector()[self.idx] = 1
#             u_obs = Function(self.out_obj.V)
#             u_obs.vector()[self.idx] = self.obs
#             fun = 0.5*self.prec*(inner(u,f_ind)-u_obs)**2
#             return fun
#         def form(self,u):
#             if type(u) is not ufl.indexed.Indexed:
#                 print('Warning: use split() instead of .split(True) to get u!')
#             # define point (Dirac) measure centered at observation locations, but point integral is limited to CG1
#             # error when compiling: Expecting test and trial spaces to only have dofs on vertices for point integrals.
#             pts_domain = VertexFunction("size_t", self.out_obj.mesh, 0) # limited to vertices, TODO: generalize to e.g. dofs nodal points
# #             pts_nbhd = AutoSubDomain(lambda x: any([near(x[0],p[0]) and near(x[1],p[1]) for p in self.loc]))
#             pts_nbhd = AutoSubDomain(lambda x: any([Point(x).distance(Point(p))<2*DOLFIN_EPS for p in self.loc]))
#             pts_nbhd.mark(pts_domain, 1)
#             self.dpm = dP(subdomain_data=pts_domain)
#             # u_obs function with observation values supported on observation locations
#             u_obs = Function(self.out_obj.V)
#             u_obs.vector()[self.idx] = self.obs
#             fom = 0.5*self.prec*(u-u_obs)**2*self.dpm(1)
#             return fom
        def ptsrc(self,u,ord=1):
            u_vec = self.extr_sol_vec(u)
            # define PointSource similar to boundary function, but PointSource is applied to (rhs) vector and is limited to scalar FunctionSpace
            dfun_vec = u_vec
            if ord==1:
                dfun_vec -= self.obs
            dfun_vec *= self.prec
            dirac = [PointSource(self.out_obj.W.sub(0),Point(p),f) for (p,f) in zip(self.loc,dfun_vec)] # fails in 1.6.0 (mac app) possibly due to swig bug in numpy.i (already fixed in numpy 1.10.2) of the system numpy 1.8.0rc1
            return dirac
#         def ptsrc1(self,u,ord=1):
#             if type(u) is not ufl.indexed.Indexed:
#                 print('Warning: use split() instead of .split(True) to get u!')
#             # define PointSource similar to boundary function, but PointSource is applied to (rhs) vector and is limited to scalar FunctionSpace
#             dfun = u
#             if ord==1:
#                 u_obs = Function(self.out_obj.V)
#                 u_obs.vector()[self.idx] = self.obs
#                 dfun -= u_obs
#             dfun *= self.prec
#             dirac = [PointSource(self.out_obj.W.sub(0),Point(p),dfun[0](list(p))) for p in self.loc]
#             return dirac
        def dirac(self,u,ord=1):
            u_vec = self.extr_sol_vec(u)
            dfun_vec = u_vec#[self.idx_dirac_rel2V]
            if ord==1:
                dfun_vec -= self.obs
            dfun_vec *= self.prec
            return dfun_vec,self.idx_dirac_global

    def data_misfit(self,obs,prec,idx=None,loc=None):
        return self._data_misfit(self,obs,prec,idx,loc)

    def get_misfit(self,obj):
        # solve forward equations
        u,_ = self.soln_fwd()
        # evaluate data-misfit function
        misfit = obj.eval(u)
        
        return misfit
    
    def soln_adj(self,obj):
        self.states_adj = Function(self.W) # adjoint states
        # Solve adjoint PDE < adj_dFdstates, states_adj > = dJdstates
#         solve(self.adj_dFdstates == self.dJdstates , self.states_adj, self.adj_bcs)
#         A,b = assemble_system(self.adj_dFdstates, self.dJdstates, self.adj_bcs)
#         solve(A, self.states_adj.vector(), b)
#         self.adj_dFdstates_assemb = PETScMatrix(); dJdstates_assemb = PETScVector()
#         assemble_system(self.adj_dFdstates, self.dJdstates, self.adj_bcs, A_tensor=self.adj_dFdstates_assemb, b_tensor=dJdstates_assemb)
#         solve(self.adj_dFdstates_assemb, self.states_adj.vector(), dJdstates_assemb)
        # error: assemble (solve) point integral (J) has supported underlying FunctionSpace more than CG1
        # have to use PointSource? Yuk!

        self.adj_dFdstates_assemb = PETScMatrix();
        assemble(self.adj_dFdstates, tensor=self.adj_dFdstates_assemb)

        u_fwd,_ = split(self.states_fwd)
        if not has_petsc4py():
            warnings.warn('Configure df with petsc4py to run faster!')
            self.dirac_1 = obj.ptsrc(u_fwd,ord=1)
            rhs_adj = Vector(mpi_comm_world(),self.W.dim())
            [delta.apply(rhs_adj) for delta in self.dirac_1]
        else:
            rhs_adj = PETScVector(mpi_comm_world(),self.W.dim())
            val_dirac_1,idx_dirac_1 = obj.dirac(u_fwd,ord=1)
            rhs_adj.vec()[idx_dirac_1] = val_dirac_1
#             np.allclose(rhs_adj.array(),rhs_adj1.vec())

        [bc.apply(self.adj_dFdstates_assemb,rhs_adj) for bc in self.adj_bcs]

        solve(self.adj_dFdstates_assemb, self.states_adj.vector(), rhs_adj)
        self.soln_count[1] += 1
        u_adj, l_adj = split(self.states_adj)
        return u_adj, l_adj

    def get_grad(self,obj):
        # solve adjoint equations
        _,_ = self.soln_adj(obj)
        # compute the gradient of dJ/dunknown = - <states_adj, adj_dFdunknown> + (dJdunknown=0)
        g_unknown_form = -action(self.adj_dFdunknown,self.states_adj)
        g_unknown_vec = assemble(g_unknown_form)
#         g_unknown = Function(self.V)
#         g_unknown.vector()[:] = g_unknown_vec
#         plot(g_unknown, title='gradient', rescale=True)
#         interactive()
        # the desired gradient dJ/dtheta = dunknown/dtheta * dJ/dunknown
        g_theta = self.dunknowndtheta_mat.T.dot(g_unknown_vec)
#         g_theta1 = np.array([assemble(action(g_unknown_form,f)) for f in self.dunknowndtheta.split()])

        return g_theta

    def soln_fwd2(self,u_actedon):
        if type(u_actedon) is np.ndarray:
            u = Function(self.V)
#             u.vector()[:] = self.dunknowndtheta_mat.dot(u_actedon)
#             dunknowndtheta_spsT = self.dunknowndtheta_sps
#             u.vector()[:] = dunknowndtheta_spsT.toarray().dot(u_actedon)
            u.vector()[:] = self.dunknowndtheta_sps.dot(u_actedon)
            u_actedon = u

        self.states_fwd2 = Function(self.W) # 2nd forward states
        # Solve 2nd forward PDE < dFdstates, states_fwd2 > = < dFdunknown, u_actedon >
#         solve(self.dFdstates == action(self.dFdunknown, u_actedon), self.states_fwd2, self.adj_bcs) # ToDo: check the boundary for fwd2
#         A,b = assemble_system(self.dFdstates, action(self.dFdunknown, u_actedon), self.adj_bcs)
#         solve(A, self.states_fwd2.vector(), b)

#         if not hasattr(self, 'dFdstates_assemb'):
#             self.dFdstates_assemb = PETScMatrix()
#             assemble(self.dFdstates, tensor=self.dFdstates_assemb)
#             [bc.apply(self.dFdstates_assemb) for bc in self.adj_bcs]

        rhs_fwd2 = PETScVector()
#         assemble(action(self.dFdunknown, u_actedon), tensor=rhs_fwd2)
        self.dFdunknown_assemb.mult(u_actedon.vector(),rhs_fwd2)

        [bc.apply(rhs_fwd2) for bc in self.adj_bcs]

        solve(self.dFdstates_assemb, self.states_fwd2.vector(), rhs_fwd2)
        self.soln_count[2] += 1
        u_fwd2, l_fwd2 = split(self.states_fwd2)
        return u_fwd2, l_fwd2

    def soln_adj2(self,obj):
        self.states_adj2 = Function(self.W) # 2nd forward states
        # Solve 2nd adjoint PDE < adj_dFdstates, states_adj2 > = < d2Jdstates, states_fwd2 >
#         solve(self.adj_dFdstates == action(self.d2Jdstates, self.states_fwd2), self.states_adj2, self.adj_bcs)
#         A,b = assemble_system(self.adj_dFdstates, action(self.d2Jdstates, self.states_fwd2), self.adj_bcs)
#         solve(A, self.states_adj2.vector(), b)

#         rhs_adj2 = PETScVector()
#         assemble(action(self.d2Jdstates, self.states_fwd2), tensor=rhs_adj2)

        u_fwd2,_ = split(self.states_fwd2)
        if not has_petsc4py():
            warnings.warn('Configure df with petsc4py to run faster!')
            self.dirac_2 = obj.ptsrc(u_fwd2,ord=2)
            rhs_adj2 = Vector(mpi_comm_world(),self.W.dim())
            [delta.apply(rhs_adj2) for delta in self.dirac_2]
        else:
            rhs_adj2 = PETScVector(mpi_comm_world(),self.W.dim())
            val_dirac_2,idx_dirac_2 = obj.dirac(u_fwd2,ord=2)
            rhs_adj2.vec()[idx_dirac_2] = val_dirac_2
#             np.allclose(rhs_adj2.array(),rhs_adj12.vec())

        [bc.apply(rhs_adj2) for bc in self.adj_bcs]

        solve(self.adj_dFdstates_assemb, self.states_adj2.vector(), rhs_adj2)
        self.soln_count[3] += 1
        u_adj2, l_adj2 = split(self.states_adj2)
        return u_adj2, l_adj2

    def get_metact(self,obj,u_actedon):
        # solve 2nd forward/adjoint equations
        _,_ = self.soln_fwd2(u_actedon)
        _,_ = self.soln_adj2(obj)
        # compute the metric action on u_actedon of d2J/dunknown = < adj_dFdunknown, states_adj2 >
#         Ma_unknown_form = action(self.adj_dFdunknown,self.states_adj2)
#         Ma_unknown_vec = assemble(Ma_unknown_form)
        Ma_unknown_vec = PETScVector()
        self.adj_dFdunknown_assemb.mult(self.states_adj2.vector(),Ma_unknown_vec)
        # the desired metric action d2J/dtheta = dunknown/dtheta * d2J/dunknown
#         Ma_theta = self.dunknowndtheta_mat.dot(Ma_unknown_vec)
#         dunknowndtheta_spsT = self.dunknowndtheta_sps
#         Ma_theta = dunknowndtheta_spsT.toarray().T.dot(Ma_unknown_vec)
        Ma_theta = self.dunknowndtheta_sps.T.dot(Ma_unknown_vec)

        return Ma_theta

    def get_met(self,obj):
        """
        Get metric by solving adjoints with multiple RHS' simultaneously-- about 10 times faster than using get_metact on unit vectors
        """

        #-- All 2nd forward states --#
        # Solve 2nd forward PDE < dFdstates, states_fwd2s > = < dFdunknown, I >
        dFdunknown_sps = petscmat2csr(self.dFdunknown_assemb)
        rhs_fwd2s_sps = dFdunknown_sps.dot(self.dunknowndtheta_sps) # multiple rhs_fwd2s

#         rhs_fwd2s_petsc = PETScMatrix(csr2petscmat(rhs_fwd2s_sps))
#         [bc.apply(rhs_fwd2s_petsc) for bc in self.adj_bcs]
#         [bc.zero(rhs_fwd2s_petsc) for bc in self.adj_bcs] # neither works

        ### apply homogenized boundary condition to the right hand side vectors using Scipy sparse matrix ###

        # homogenized Dirichlet boundary condition
        for bc in self.adj_bcs:
            binds = bc.get_boundary_values().keys()
            csr_zero_rows(rhs_fwd2s_sps,binds)

        states_fwd2s_sps = sps.linalg.spsolve(petscmat2csr(self.dFdstates_assemb),rhs_fwd2s_sps) # csc? # all 2nd forward states simultaneously
        self.soln_count[2] += states_fwd2s_sps.shape[1]

        #-- All 2nd adjoint states --#
        # Solve 2nd adjoint PDE < adj_dFdstates, states_adj2s > = < d2Jdstates, states_fwd2s >
#         rhs_adj2s_sps = sps.csr_matrix(states_fwd2s_sps.shape)
#         rhs_adj2s_sps[obj.idx_dirac,] = obj.prec*states_fwd2s_sps[obj.idx_dirac,]

        rhs_adj2s_sps = obj.prec*states_fwd2s_sps # multiple rhs_adj2s
        csr_keep_rows(rhs_adj2s_sps,obj.idx_dirac_global) # using global dof index to avoid Point Source
#         csr_zero_rows(rhs_adj2s_sps,np.setdiff1d(range(states_fwd2s_sps.shape[0]),obj.idx_dirac))

        # homogenized Dirichlet boundary condition
        for bc in self.adj_bcs:
            binds = bc.get_boundary_values().keys()
            csr_zero_rows(rhs_adj2s_sps,binds)

        states_adj2s_sps = sps.linalg.spsolve(petscmat2csr(self.adj_dFdstates_assemb),rhs_adj2s_sps) # csc? # all 2nd adjoint states simultaneously
        self.soln_count[3] += states_adj2s_sps.shape[1]

        # compute the metric action on u_actedon of d2J/dunknown = < adj_dFdunknown, states_adj2 >
        M_unknown_sps = petscmat2csr(self.adj_dFdunknown_assemb).dot(states_adj2s_sps)
        # the desired metric action d2J/dtheta = dunknown/dtheta * d2J/dunknown
#         M_unknown_mat = M_unknown_sps
#         M_theta = self.dunknowndtheta_mat.dot(M_unknown_mat.toarray())
        M_theta = self.dunknowndtheta_sps.T.dot(M_unknown_sps)
        M_theta = M_theta.toarray()

        return M_theta

    def get_geom(self,coeff,obj=None,geom_ord=[0],trt_opt=-1,trt_idx=[],log_level=ERROR):
        val=None; grad=None; metact=None; met=None;
        # set log level: DBG(10), TRACE(13), PROGRESS(16), INFO(20,default), WARNING(30), ERROR(40), or CRITICAL(50)
        set_log_level(log_level)
        if not (any(trt_idx) or 0<=trt_opt<=2):
            trt_idx = range(coeff.l) # trt_idx: indices of parameters' components left
        if trt_opt==0:
            coeff.theta = coeff.theta[trt_idx] # trt_opt: the derivative order where truncation starts: 2 only for metric; 1 for both gradient and metric; 0 for all geometrics
            coeff.l = len(trt_idx)
        self.set_forms(coeff=coeff,obj=obj,geom_ord=geom_ord,trt_opt=trt_opt,trt_idx=trt_idx)
#         import time
#         start=time.time()
        if any(s>=0 for s in geom_ord):
            u,_ = self.soln_fwd()
            val = obj.eval(u)
#         end=time.time()
#         print('Time for obtaining misfit value is %.10f' % (end-start))
#         start=time.time()
        if any(s>=1 for s in geom_ord):
            grad = self.get_grad(obj)
#         end=time.time()
#         print('Time for obtaining the gradient is %.10f' % (end-start))
#         start=time.time()
        if any(s>1 for s in geom_ord):
            # do some assembling here to avoid repetition
            # for fwd2:
            self.dFdstates_assemb = PETScMatrix()
            assemble(self.dFdstates, tensor=self.dFdstates_assemb)
            [bc.apply(self.dFdstates_assemb) for bc in self.adj_bcs]
            self.dFdunknown_assemb = PETScMatrix()
            assemble(self.dFdunknown, tensor=self.dFdunknown_assemb)
            # for metact:
            self.adj_dFdunknown_assemb = PETScMatrix()
            assemble(self.adj_dFdunknown, tensor=self.adj_dFdunknown_assemb)
        if 1.5 in geom_ord:
            metact = lambda v: self.get_metact(obj,v)
        if 2 in geom_ord:
            if has_petsc4py():
#                 import time
#                 start=time.time()
                met = self.get_met(obj)
#                 end=time.time()
#                 print('k! times is %.10f' % (end-start))
            else:
                warnings.warn('Configure df with petsc4py to run faster!')
                metact = lambda v: self.get_metact(obj,v)
#                 start=time.time()
                met = np.array([metact(e) for e in np.eye(self.dunknowndtheta_sps.shape[1])])
#                 end=time.time()
#                 print('o! times is %.10f' % (end-start))
#         end=time.time()
#         print('Time for obtaining the metric is %.10f' % (end-start))
        return val,grad,metact,met

    def get_geom_(self,coeff,obj=None,ord=[0],trt_opt=-1,trt_idx=[],log_level=ERROR,**kwargs):
        loglik=None; agrad=None; metact=None; met=None; eigs=None
        # set log level: DBG(10), TRACE(13), PROGRESS(16), INFO(20,default), WARNING(30), ERROR(40), or CRITICAL(50)
        set_log_level(log_level)
        if not (any(trt_idx) or 0<=trt_opt<=2):
            trt_idx = range(coeff.l) # trt_idx: indices of parameters' components left
        if trt_opt==0:
            coeff.theta = coeff.theta[trt_idx] # trt_opt: the derivative order where truncation starts: 2 only for metric; 1 for both gradient and metric; 0 for all geometrics
            coeff.l = len(trt_idx)
        self.set_forms(coeff=coeff,obj=obj,geom_ord=ord,trt_opt=trt_opt,trt_idx=trt_idx)
#         import time
#         start=time.time()
        if any(s>=0 for s in ord):
            loglik = -self.get_misfit(obj)
#         end=time.time()
#         print('Time for obtaining misfit value is %.10f' % (end-start))
#         start=time.time()
        if any(s>=1 for s in ord):
            agrad = np.zeros(coeff.l)
            if trt_opt==1:
                agrad[trct_idx] -= self.get_grad(obj)
            else:
                agrad -= self.get_grad(obj)
#         end=time.time()
#         print('Time for obtaining the gradient is %.10f' % (end-start))
#         start=time.time()
        if any(s>1 for s in ord):
            # do some assembling here to avoid repetition
            # for fwd2:
            self.dFdstates_assemb = PETScMatrix()
            assemble(self.dFdstates, tensor=self.dFdstates_assemb)
            [bc.apply(self.dFdstates_assemb) for bc in self.adj_bcs]
            self.dFdunknown_assemb = PETScMatrix()
            assemble(self.dFdunknown, tensor=self.dFdunknown_assemb)
            # for metact:
            self.adj_dFdunknown_assemb = PETScMatrix()
            assemble(self.adj_dFdunknown, tensor=self.adj_dFdunknown_assemb)
        if 1.5 in ord:
            metact = lambda v: self.get_metact(obj,v)
        if 2 in ord:
            if has_petsc4py():
#                 import time
#                 start=time.time()
                met = self.get_met(obj)
#                 end=time.time()
#                 print('k! times is %.10f' % (end-start))
            else:
                warnings.warn('Configure df with petsc4py to run faster!')
                metact = lambda v: self.get_metact(obj,v)
#                 start=time.time()
                met = np.array([metact(e) for e in np.eye(self.dunknowndtheta_sps.shape[1])])
#                 end=time.time()
#                 print('o! times is %.10f' % (end-start))
#         end=time.time()
#         print('Time for obtaining the metric is %.10f' % (end-start))
            # adjust the gradient
            agrad[trt_idx]+=met.dot(coeff.theta[trt_idx])
            # compute eigen-decomposition TODO: use randomized algorithms
            if 'k' in kwargs:
                k = min([kwargs['k'],met.shape[0]-1])
                eigs = sps.linalg.eigsh(met+np.eye(met.shape[0]),k)
        
        if not any(kwargs):
            return loglik,agrad,metact,met
        else:
            return loglik,agrad,metact,eigs
    
    def save_soln(self,sep=False):
        # title settings
        self.titles = ['Potential Function','Lagrange Multiplier']
        self.sols = ['fwd','adj','fwd2','adj2']
        self.sub_titles = ['forward','adjoint','2nd forward','2nd adjoint']
        import os
        cwd=os.getcwd()
        self.savepath=os.path.join(cwd,'result')
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
        for i,sol in enumerate(self.sols):
            # get solution
            sol_name = '_'.join(['states',sol])
            try:
                soln = getattr(self,sol_name)
            except AttributeError:
                print(self.sub_titles[i]+'solution not found!')
                pass
            else:
                if not sep:
                    File(os.path.join(self.savepath,sol_name+'.xml'))<<soln
                else:
                    soln = soln.split(True)
                    for j,splt in enumerate(self.titles):
                        File(os.path.join(self.savepath,'_'.join([splt,sol])+'.pvd'))<<soln[j]

    def _plot_vtk(self,SAVE=False):
        for i,sol in enumerate(self.sols):
            # get solution
            try:
                soln = getattr(self,'_'.join(['states',sol]))
            except AttributeError:
                print(self.sub_titles[i]+'solution not found!')
                pass
            else:
                soln = soln.split(True)
                for j,titl in enumerate(self.titles):
                    fig=plot(soln[j],title=self.sub_titles[i]+' '+titl,rescale=True)
                    if SAVE:
                        fig.write_png(os.path.join(self.savepath,'_'.join([titl,sol])+'.png'))

    def _plot_mpl(self,SAVE=False):
        import matplotlib.pyplot as plt
        try:
            col_bar_supp = True
            parameters["plotting_backend"]="matplotlib"
        except KeyError:
            col_bar_supp = False #no colorbar support on older version
            print('Warning: plot has not been overloaded with matplotlib before version 1.7.0-dev!')
#             from util import matplot4dolfin
            if SAVE:
                savepath=self.savepath
            else:
                savepath=None
            matplot=matplot4dolfin(SAVE=SAVE,savepath=savepath)
            global plot
            plot=matplot.mplot
        # codes for plotting solutions
        import matplotlib as mp
        for i,titl in enumerate(self.titles):
            fig,axes = plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True,num=i,figsize=(10,6))
            for j,ax in enumerate(axes.flat):
                # get solution
                try:
                    soln = getattr(self,'_'.join(['states',self.sols[j]]))
                except AttributeError:
                    print(self.sub_titles[j]+'solution not found!')
                    pass
                else:
                    soln = soln.split(True)
                    plt.axes(ax)
                    sub_fig = plot(soln[i])
                    plt.axis([0, 1, 0, 1])
                    ax.set_title(self.sub_titles[j])
            if col_bar_supp:
                cax,kw = mp.colorbar.make_axes([ax for ax in axes.flat])
                plt.colorbar(sub_fig, cax=cax, **kw) # TODO: fix the issue of common color range
            # set common titles
            fig.suptitle(titl)
            # tight layout
            plt.tight_layout()
            if SAVE:
                matplot.savefig(titl+'.png',bbox_inches='tight')

    def plot_soln(self,backend='matplotlib',SAVE=False):
#         parameters["plotting_backend"]=backend
        # title settings
        if not hasattr(self, 'titles'):
            self.titles = ['Potential Function','Lagrange Multiplier']
        if not hasattr(self, 'sols'):
            self.sols = ['fwd','adj','fwd2','adj2']
        if not hasattr(self, 'sub_titles'):
            self.sub_titles = ['forward','adjoint','2nd forward','2nd adjoint']
        if SAVE:
            import os
            if not hasattr(self, 'savepath'):
                cwd=os.getcwd()
                self.savepath=os.path.join(cwd,'result')
                if not os.path.exists(self.savepath):
                    print('Save path does not exist; created one.')
                    os.makedirs(self.savepath)
        if backend is 'matplotlib':
            import matplotlib.pyplot as plt
            self._plot_mpl(SAVE=SAVE)
            plt.show()
        elif backend is 'vtk':
            self._plot_vtk(SAVE=SAVE)
            interactive()
        else:
            raise Exception(backend+'not found!')

    def test(self,SAVE=False,PLOT=False,dim_theta=9,kl_opt='fb',sigma=1,s=1.1,SNR=10,chk_fd=False,h=1e-4,**kwargs):
        np.random.seed(2016)
        # generate theta
        theta=np.random.randn(dim_theta)
        coeff=self.coefficient(theta=theta,kl_opt=kl_opt,sigma=sigma,s=s,degree=2,**kwargs)
        # obtain observations
        obs,idx,loc,sd_noise=self.get_obs(unknown=coeff,SNR=SNR)
        # define data misfit class
        print('\nDefining data-misfit...')
        misfit=self.data_misfit(obs,1./sd_noise**2,idx,loc)

        import time
        # obtain the geometric quantities
        print('\n\nObtaining geometric quantities with Adjoint method...')
        start = time.time()
        loglik,grad,_,_ = self.get_geom_(coeff,misfit,[0,1])
        _,_,_,FI = self.get_geom_(coeff,misfit,[2])
        end = time.time()
        print('Time used is %.4f' % (end-start))

        # save solutions to file
        if SAVE:
            self.save_soln()
        # plot solutions
        if PLOT:
            self.plot_soln()

        if chk_fd:
            # check with finite difference
            print('\n\nTesting against Finite Difference method...')
            start = time.time()
            # random direction
            v = np.random.randn(coeff.l)
            ## gradient
            print('\nChecking gradient:')
            theta_p = theta.copy(); theta_p += h*v
            coeff.theta=theta_p # update theta
            loglik_p,_,_,_ = self.get_geom_(coeff,misfit)
            theta_m = theta.copy(); theta_m -= h*v
            coeff.theta=theta_m # update theta
            loglik_m,_,_,_ = self.get_geom_(coeff,misfit)
            dloglikv_fd = (loglik_p-loglik_m)/(2*h)
            dloglikv = grad.dot(v)
            rdiff_gradv = np.abs(dloglikv_fd-dloglikv)/np.linalg.norm(v)
            print('Relative difference of gradients in a random direction between adjoint and finite difference: %.10f' % rdiff_gradv)

            # random direction
            w = np.random.randn(coeff.l)
            ## metric-action
            print('\nChecking Metric-action:')
            dgradvw_fd = 0
            # obtain sensitivities
            for n in range(len(idx)):
                misfit_n=self.data_misfit(obs[n],1./sd_noise**2,idx[n],loc[None,n,])
                # in direction v
                theta_p = theta.copy(); theta_p += h*v
                coeff.theta=theta_p; self.set_forms(coeff)
                u_p,_ = self.soln_fwd()
                u_p_v = misfit_n.extr_sol_vec(u_p)
                theta_m = theta.copy(); theta_m -= h*v
                coeff.theta=theta_m; self.set_forms(coeff)
                u_m,_ = self.soln_fwd()
                u_m_v = misfit_n.extr_sol_vec(u_m)
                dudtheta_v=(u_p_v-u_m_v)/(2*h)
                # in direction w
                theta_p = theta.copy(); theta_p += h*w
                coeff.theta=theta_p; self.set_forms(coeff)
                u_p,_ = self.soln_fwd()
                u_p_w = misfit_n.extr_sol_vec(u_p)
                theta_m = theta.copy(); theta_m -= h*w
                coeff.theta=theta_m; self.set_forms(coeff)
                u_m,_ = self.soln_fwd()
                u_m_w = misfit_n.extr_sol_vec(u_m)
                dudtheta_w=(u_p_w-u_m_w)/(2*h)
                # Metric (Gauss-Newton Hessian) with one observation
                dgradvw_fd += dudtheta_w*dudtheta_v
            dgradvw_fd *= misfit.prec
            dgradvw = w.dot(FI.dot(v))
#             dgradvw = w.dot(Fv(v))
            rdiff_Metvw = np.abs(dgradvw_fd-dgradvw)/np.linalg.norm(v)/np.linalg.norm(w)
            print('Relative difference of Metrics in two random directions between adjoint and finite difference: %.10f' % rdiff_Metvw)
            end = time.time()
            print('Time used is %.4f' % (end-start))

# class _get_nested():
#     """
#     When called with the containing class as the first argument, 
#     and the name of the nested class as the second argument,
#     returns an instance of the nested class.
#     http://stackoverflow.com/questions/1947904/how-can-i-pickle-a-nested-class-in-python
#     """
#     def __call__(self,container,nested):
#         nested_class=getattr(container, nested)
#         # make an instance of a simple object (this one will do), for which we can change the __class__ later on.
#         nested_instance=_get_nested()
#         # set the class of the instance, the __init__ will never be called on the class
#         # but the original state will be set later on by pickle.
#         nested_instance.__class__ = nested_class
#         return nested_instance

if __name__ == '__main__':
    elliptic = Elliptic(nx=40,ny=40)
    elliptic.test(SAVE=False,PLOT=False,dim_theta=100,kl_opt='fb',chk_fd=True,h=1e-5)
