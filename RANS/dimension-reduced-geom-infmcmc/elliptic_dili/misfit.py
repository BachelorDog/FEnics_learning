#!/usr/bin/env python
"""
Class definition of data-misfit for Elliptic PDE model in the DILI paper by Cui et~al (2016).
Data come from file or are generated according to the model in a mesh finer than the one for inference.
------------------------------------------------------------
written in FEniCS 2016.2.0-dev, with backward support for 1.6.0
Shiwei Lan @ U of Warwick, 2016; @ Caltech, Sept. 2016
-------------------------------
Created July 30, 2016
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2016, The EQUIP/EQUiPS projects"
__license__ = "GPL"
__version__ = "0.7"
__maintainer__ = "Shiwei Lan"
__email__ = "S.Lan@warwick.ac.uk; slan@caltech.edu; lanzithinking@outlook.com"

import dolfin as df
import ufl
import numpy as np

# self defined modules
from pde import ellipticPDE
# from prior import Gaussian_prior
import sys
sys.path.append( "../" )
from util.dolfin_gadget import *

# true transmissivity field
class _true_coeff(df.Expression):
    """
    True transmissivity filed, the coefficient of elliptic PDE used for generating observations.
    """
    def __init__(self,**kwargs):
        self.truth_area1 = lambda x: .6<= x[0] <=.8 and .2<= x[1] <=.4
        self.truth_area2 = lambda x: (.8-.3)**2<= (x[0]-.8)**2+(x[1]-.2)**2 <=(.8-.2)**2 and x[0]<=.8 and x[1]>=.2
    def eval(self,value,x):
        if self.truth_area1(x) or self.truth_area2(x):
            value[0] = -1
        else:
            value[0] = 0
    def plot(self,V):
        u_coeff=df.interpolate(self, V)
        from util import matplot4dolfin
        matplot=matplot4dolfin(overloaded=False) # switch overloaded to use the default or the manually coded plotting function
        fig=matplot.plot(u_coeff)
#         matplot.show()
        return fig

# def _get_sd_noise(prior,SNR=10,num_priors=100):
#     """
#     Get standard deviation of noise based on Signal-Noise-Ratio, SNR:=max{u}/sd_noise, where u~prior
#     """
#     u_max=0
#     for n in range(num_priors):
#         u_max+=prior.sample().max()
#     u_max/=num_priors
#     sd_noise=u_max/SNR
#     return sd_noise
    
def get_obs(pde4inf=None,unknown=None,SNR=10,SAVE=False):
    """
    Read observations from the file or generate them by solving elliptic PDE on a fine mesh.
    pde4inf: PDE model defined on a coarser mesh for inference.
    unknown: unknown function used for generating data.
    SNR: Signal-Noise-Ratio: max(|p|)/sd(noise)
    Output:
    obs: observation values, PDE solutions at selected locations contaminated with noise.
    idx,loc: dof indices and coordinate locations of observations in the solution domain.
    sd_noise: standard noise of the noise added to PDE solutions.
    """
    import os,pickle
    f_obs=os.path.join(os.getcwd(),'obs_SNR'+str(SNR)+'.pckl')
    if os.path.isfile(f_obs) and unknown is None:
        f=open(f_obs,'rb')
        obs,idx,loc,sd_noise=pickle.load(f)
        f.close()
        # update indices, locations and observations on the FEM for inference
        idx,loc,rel_idx = check_in_dof(loc,pde4inf.V,tol=1e-6)
        obs = obs[rel_idx]
        print('%d observations have been read!' % len(idx))
    else:
        print('Obtaining observations on a refined (e.g. double-sized) mesh...')
        if pde4inf is None:
            pde4inf = ellipticPDE()
        # define the model on a finer mesh to obtain observations
        pde4obs = ellipticPDE(nx=2*pde4inf.nx,ny=2*pde4inf.ny)
        if unknown is None:
#             unknown = _true_coeff(degree=0)
            unknown = df.interpolate(_true_coeff(degree=0), pde4obs.V)
#         if prior is None:
#             prior = Gaussian_prior(V=pde4obs.V,mpi_comm=pde4obs.mesh.mpi_comm())
        # obtain the solution of u on a finer mesh
        if unknown.function_space()!=pde4obs.V:
            unknown = df.interpolate(unknown, pde4obs.V)
        pde4obs.set_forms(unknown)
        _,_=pde4obs.soln_fwd()
        u,_=pde4obs.states_fwd.split(True)
        u_vec = u.vector()
        # choose locations based on a set of coordinates
        sq = np.arange(.2,.6+.1,.1)
        X, Y = np.meshgrid(sq, sq)
        loc = np.array([X.flatten(),Y.flatten()]).T
        # check those locations against solution mesh
        idx,loc,_ = check_in_dof(loc,pde4obs.V,tol=1e-6) # dof index in V
        # obtain observations by adding noise to these solutions
        if idx is not None:
            sol_on_loc = u_vec[idx]
        else:
            sol_on_loc = [u(list(x)) for x in loc]
        # obtain standard deviation of noise
#         sd_noise = _get_sd_noise(prior,SNR=SNR)
        u_max = np.max(np.abs([unknown.vector().min(),unknown.vector().max()]))
        sd_noise = u_max/SNR
        obs = sol_on_loc + sd_noise*np.random.randn(len(sol_on_loc))
        # revert to the original coarser mesh for inference
        # update indices, locations and observations
        idx,loc,rel_idx = check_in_dof(loc,pde4inf.V,tol=1e-6)
        obs = obs[rel_idx]
        print('%d observations have been obtained!' % len(idx))
        # save observations to file
        if SAVE:
            f=open(f_obs,'wb')
            pickle.dump([obs,idx,loc,sd_noise],f)
            f.close()
    return obs,sd_noise,idx,loc

class data_misfit(object):
    """
    Class definition of data-misfit function
    misfit := int (y-Ou)'prec(y-Ou) dx,
    with u: PDE solutions, y: observations, prec: precision of observational noise, and O: observational operator.
    Input: PDE model, observation values, precision of observational noise, dof indices or coordinate locations of observations.
    Output: dof indices relative to the mixed function space, and methods to provide right hand sides of forward/adjoint equations.
    """
    def __init__(self,pde,obs=None,sd_noise=None,idx=None,loc=None,**kwargs):
        """
        Initialize data-misfit class with information of observations.
        """
        self.pde = pde
        if None in [obs,sd_noise,idx] or None in [obs,sd_noise,loc]:
            obs,sd_noise,idx,loc=get_obs(pde4inf=pde,SAVE=True,**kwargs)
        self.obs = obs
        self.prec = 1.0/sd_noise**2
        self.idx = idx
        self.loc = loc
        self.SNR = kwargs.pop('SNR',10)
#         # define point (Dirac) measure centered at observation locations, but point integral is limited to CG1
#         # error when compiling: Expecting test and trial spaces to only have dofs on vertices for point integrals.
#         pts_domain = df.VertexFunction("size_t", self.pde.mesh, 0) # limited to vertices, TODO: generalize to e.g. dofs nodal points
# #             pts_nbhd = df.AutoSubDomain(lambda x: any([near(x[0],p[0]) and near(x[1],p[1]) for p in self.loc]))
#         pts_nbhd = df.AutoSubDomain(lambda x: any([df.Point(x).distance(df.Point(p))<2*df.DOLFIN_EPS for p in self.loc]))
#         pts_nbhd.mark(pts_domain, 1)
#         self.dpm = df.dP(subdomain_data=pts_domain)
        # find global dof of observations
        if self.idx is None:
            idx_dirac_local,_,self.idx_dirac_rel2V = check_in_dof(self.loc, self.pde.V) # idx_dirac_rel2Vv: indices relative to V
        else:
            idx_dirac_local = self.idx # indices relative to V
        sub_dofs = self.pde.W.sub(0).dofmap().dofs() # dof map: V --> W
        self.idx_dirac_global = sub_dofs[idx_dirac_local] # indices relative to W
    def _extr_soloc(self,u):
        """
        Return solution (u) values at the observational locations.
        """
        # u_vec: solution vector at observation locations
        if type(u) is ufl.indexed.Indexed:
            u_vec = [u(list(x)) for x in self.loc]
        elif type(u) is df.functions.function.Function:
            if self.idx is not None:
                u_vec = u.vector()[self.idx]
            elif self.loc is not None:
                u_vec = [u(list(x))[0] for x in self.loc]
        elif type(u) is df.cpp.la.GenericVector or np.ndarray:
            u_vec = u[self.idx]
        else:
            raise Exception('Check the type of u! Either the indeces or the locations of observations are needed!')
        return np.array(u_vec)
    def eval(self,u):
        """
        Evaluate misfit function for given solution u.
        """
        u_vec = self._extr_soloc(u)
        diff = u_vec-self.obs
        val = 0.5*self.prec*diff.dot(diff)
        return val
#     def func(self,u):
#         if type(u) is not ufl.indexed.Indexed:
#             print('Warning: use split() instead of .split(True) to get u!')
#         f_ind = df.Function(self.pde.V)
# #         f_ind.vector()[:] = 0
#         f_ind.vector()[self.idx] = 1
#         u_obs = df.Function(self.pde.V)
#         u_obs.vector()[self.idx] = self.obs
#         fun = 0.5*self.prec*(inner(u,f_ind)-u_obs)**2
#         return fun
#     def form(self,u):
#         if type(u) is not ufl.indexed.Indexed:
#             print('Warning: use split() instead of .split(True) to get u!')
#         # define point (Dirac) measure centered at observation locations, but point integral is limited to CG1
#         # error when compiling: Expecting test and trial spaces to only have dofs on vertices for point integrals.
#         pts_domain = df.VertexFunction("size_t", self.pde.mesh, 0) # limited to vertices, TODO: generalize to e.g. dofs nodal points
# #         pts_nbhd = df.AutoSubDomain(lambda x: any([near(x[0],p[0]) and near(x[1],p[1]) for p in self.loc]))
#         pts_nbhd = df.AutoSubDomain(lambda x: any([df.Point(x).distance(df.Point(p))<2*df.DOLFIN_EPS for p in self.loc]))
#         pts_nbhd.mark(pts_domain, 1)
#         self.dpm = df.dP(subdomain_data=pts_domain)
#         # u_obs function with observation values supported on observation locations
#         u_obs = df.Function(self.pde.V)
#         u_obs.vector()[self.idx] = self.obs
#         fom = 0.5*self.prec*(u-u_obs)**2*self.dpm(1)
#         return fom
    def ptsrc(self,u,ord=1):
        """
        Point source of (ord) order derivative of data-misfit function wrt. the solution u.
        """
        assert ord in [1,2], 'Wrong order!'
        u_vec = self._extr_soloc(u)
        # define PointSource similar to boundary function, but PointSource is applied to (rhs) vector and is limited to scalar FunctionSpace
        dfun_vec = u_vec
        if ord==1:
            dfun_vec -= self.obs
        dfun_vec *= self.prec
        dirac = [df.PointSource(self.pde.W.sub(0),df.Point(p),f) for (p,f) in zip(self.loc,dfun_vec)] # fails in 1.6.0 (mac app) possibly due to swig bug in numpy.i (already fixed in numpy 1.10.2) of the system numpy 1.8.0rc1
        return dirac
#     def ptsrc1(self,u,ord=1):
#         if type(u) is not ufl.indexed.Indexed:
#             print('Warning: use split() instead of .split(True) to get u!')
#         # define PointSource similar to boundary function, but PointSource is applied to (rhs) vector and is limited to scalar FunctionSpace
#         dfun = u
#         if ord==1:
#             u_obs = df.Function(self.pde.V)
#             u_obs.vector()[self.idx] = self.obs
#             dfun -= u_obs
#         dfun *= self.prec
#         dirac = [df.PointSource(self.pde.W.sub(0),df.Point(p),dfun[0](list(p))) for p in self.loc]
#         return dirac
    def dirac(self,u,ord=1):
        """
        Dirac function of (ord) order derivative of data-misfit function wrt. the solution u.
        """
        assert ord in [1,2], 'Wrong order!'
        u_vec = self._extr_soloc(u)
        dfun_vec = u_vec#[self.idx_dirac_rel2V]
        if ord==1:
            dfun_vec -= self.obs
        dfun_vec *= self.prec
        return dfun_vec,self.idx_dirac_global
    
    def plot_data(self):
        """
        Plot the data information.
        """
        import matplotlib.pyplot as plt
        from util import matplot4dolfin
        matplot=matplot4dolfin()
        fig=matplot.plot(self.pde.mesh)
        plt.axis('tight')
#         plt.plot(self.loc[:,0],self.loc[:,1],'bo',markersize=10)
        plt.scatter(self.loc[:,0],self.loc[:,1],c=self.obs,s=200,zorder=2)
#         plt.xlim(0,1); plt.ylim(0,1)
#         plt.axis('tight')
#         plt.xlabel('x',fontsize=12); plt.ylabel('y',fontsize=12,rotation=0)
#         plt.title('Observations on selected locations',fontsize=12)
#         matplot.show()
        return fig
    
if __name__ == '__main__':
    np.random.seed(2017)
    # define PDE
    pde=ellipticPDE()
    # get the true parameter function
    truth = _true_coeff(degree=0)
    # define data misfit
    misfit = data_misfit(pde,SNR=10)
    # plot
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    fig,axes = plt.subplots(nrows=1,ncols=2,sharex=True,figsize=(14,5))
    sub_figs = [None]*len(axes)
    cm_clim = np.min([-1,misfit.obs.min()]),np.max([0,misfit.obs.max()])
    # plot the truth
    plt.axes(axes[0])
    sub_figs[0]=truth.plot(pde.V)
    sub_figs[0].set_clim(cm_clim)
    plt.title('True log-transmissivity field',fontsize=12)
#     ax.set_title('True transmissivity field',fontsize=12)
    # plot observations
    plt.axes(axes[1])
    sub_figs[1]=misfit.plot_data() # a bug in system matplotlib 1.3.1 to return None
    plt.clim(cm_clim)
    plt.title('Observations on selected locations',fontsize=12)
#     ax.set_title('Observations on selected locations',fontsize=12)
    cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
    plt.colorbar(sub_figs[1], cax=cax, **kw)
    # fig.tight_layout()
    plt.savefig('./result/truth_obs.png',bbox_inches='tight')
    plt.show()