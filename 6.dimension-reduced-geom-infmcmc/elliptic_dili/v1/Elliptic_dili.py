#!/usr/bin/env python
"""
Class definition of inverse Elliptic PDE problem in the DILI paper by Cui et~al (2016)
written in FEniCS 2016.2.0-dev, with backward support for 1.6.0, portable to other PDE models
Shiwei Lan @ U of Warwick, 2016; @ Caltech, Sept. 2016
-----------------------------------
The purpose of this script is to obtain geometric quantities, misfit, its gradient and the associated metric (Gauss-Newton) using adjoint methods.
--To run demo:                     python Elliptic_dili.py # to compare with the finite difference method
--To initialize problem:     e.g.  elliptic=Elliptic(args)
--To obtain geometric quantities:  loglik,agrad,metact,[eigs] = elliptic.get_geom(args) # misfit value, gradient, metric action and eigenpairs of metric resp.
                                   which calls _get_misfit, _get_grad_misfit, and _get_metact_misfit resp.
--To save PDE solutions:           elliptic.save()
                                   fwd: forward solution; adj: adjoint solution; fwd2: 2nd order forward; adj2: 2nd order adjoint.
--To plot PDE solutions:           elliptic.plot()
-----------------------------------
Created May 18, 2016
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2016, The EQUIP/EQUiPS projects"
__license__ = "GPL"
__version__ = "0.9"
__maintainer__ = "Shiwei Lan"
__email__ = "S.Lan@warwick.ac.uk; slan@caltech.edu; lanzithinking@outlook.com"

# import modules
import dolfin as df
# import ufl
import numpy as np

# self defined modules
import sys
sys.path.append( "../" )
from util import *
from pde import *
from prior import *
from misfit import *
from posterior import *

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')
# set_log_level(ERROR)

class Elliptic:
    def __init__(self,nx=40,ny=40,nugg=1.0e-20,SNR=10,**kwargs):
        """
        Initialize the inverse Elliptic problem by defining the physical PDE model, the prior model and the misfit (likelihood) model.
        """
        # set pde
        self.pde = ellipticPDE(nx=nx,ny=ny,nugg=nugg)
#         self.mpi_comm = self.pde.mpi_comm
        print('\nPhysical PDE model is defined.')
        # set prior
        self.prior = Gaussian_prior(V=self.pde.V,mpi_comm=self.pde.mpi_comm,**kwargs)
        print('\nPrior model is specified.')
        # set misfit
        self.misfit = data_misfit(self.pde,prior=self.prior,SNR=SNR)
        print('\nLikelihood model is obtained.')
        # set low-rank approximate Gaussian posterior
        self.post_Ga = Gaussian_posterior_LRapp(self.prior,eigs=None)
        print('\nApproximate posterior model is set.\n')
        
        # count PDE solving times
#         self.soln_count = np.zeros(4)
        # 0-3: number of solving (forward,adjoint,2ndforward,2ndadjoint) equations respectively
    
    def _get_misfit(self):
        """
        Get the misfit function value.
        J:= int (y-Op)'prec(y-Op) dx [= sum_n (y_n-p(x_n,))'prec_n(y_n-p(x_n,))]
        """
        # solve forward equations
        u,_ = self.pde.soln_fwd()
        # evaluate data-misfit function
        val = self.misfit.eval(u)
        
        return val
    
    def _get_grad_misfit(self):
        """
        Get the gradient of misfit function.
        dJ/dunknown = - <states_adj, adj_dFdunknown> + (dJdunknown=0)
        """
        # solve adjoint equations
        _,_ = self.pde.soln_adj(self.misfit)
        # compute the gradient of dJ/dunknown = - <states_adj, adj_dFdunknown> + (dJdunknown=0)
#         g_unknown_form = -df.action(self.pde.adj_dFdunknown,self.pde.states_adj)
#         g_unknown_vec = df.assemble(g_unknown_form)
#         g_unknown = df.Function(self.pde.V)
        g_unknown_vec = df.Vector()
        self.pde.adj_dFdunknown_assemb.init_vector(g_unknown_vec,0)
#         g_unknown.vector()[:] = g_unknown_vec
#         self.pde.adj_dFdunknown_assemb.mult(-self.pde.states_adj.vector(),g_unknown.vector())
        self.pde.adj_dFdunknown_assemb.mult(-self.pde.states_adj.vector(),g_unknown_vec)
#         np.isclose(g_unknown_vec,g_unknown.vector()).all()
#         df.plot(g_unknown, title='gradient', rescale=True)
#         df.interactive()

#         return g_unknown
        return g_unknown_vec

    def _get_metact_misfit(self,u_actedon):
        """
        Get the metric-action of misfit: a--> Ma.
        d2J/dunknown(a) = < adj_dFdunknown, states_adj2(a) >
        """
        if type(u_actedon) is not df.Function:
#             assert u_actedon.size == self.pde.V.dim(), "Metric must act on a vector of size consistent with mesh!"
            u_actedon = vec2fun(u_actedon,self.pde.V)
        
        # solve 2nd forward/adjoint equations
        _,_ = self.pde.soln_fwd2(u_actedon)
        _,_ = self.pde.soln_adj2(self.misfit)
        # compute the metric action on u_actedon of d2J/dunknown = < adj_dFdunknown, states_adj2 >
#         Ma_unknown_form = df.action(self.pde.adj_dFdunknown,self.pde.states_adj2)
#         Ma_unknown_vec = df.assemble(Ma_unknown_form)
#         Ma_unknown = df.Function(self.pde.V)
        Ma_unknown_vec = df.Vector()
        self.pde.adj_dFdunknown_assemb.init_vector(Ma_unknown_vec,0)
#         self.pde.adj_dFdunknown_assemb.mult(self.pde.states_adj2.vector(),Ma_unknown.vector())
        self.pde.adj_dFdunknown_assemb.mult(self.pde.states_adj2.vector(),Ma_unknown_vec)

#         return Ma_unknown
        return Ma_unknown_vec
    
    def get_geom(self,unknown,geom_ord=[0],whitened=False,log_level=df.ERROR,**kwargs):
        """
        Get geometric quantities of the objective function (data-misfit) required geometric MCMC.
        loglik: log-likelihood, the negative misfit function, -.5 * prec * sum_n (y_n-p(x_n,))^2
        gradlik: (adjusted) gradient of log-likelihood, - ( (prior_prec - precond) * parameter + grad_misifit )
        metact: metric (Gauss-Newton Hessian of misfit or prior pre-conditioned GNH for whitened parameter) action on vector
        eigs: first (k) eigenpairs of met.
        """
        loglik=None; gradlik=None; metact=None; eigs=None
        # set log level: DBG(10), TRACE(13), PROGRESS(16), INFO(20,default), WARNING(30), ERROR(40), or CRITICAL(50)
        df.set_log_level(log_level)
        
        # convert unknown to function when necessary
#         if type(unknown) is not df.Function:
#             unknown = vec2fun(unknown,self.pde.V)
        
        # set (and assemble) forms
#         self.pde.set_forms(unknown=unknown,geom_ord=geom_ord)
        self.pde.set_forms(unknown=vec2fun(unknown,self.pde.V),geom_ord=geom_ord)
#         self.pde.assemble_forms(geom_ord=geom_ord)
        
        if any(s>=0 for s in geom_ord):
            loglik = -self._get_misfit()
        
        # assemble forms
        self.pde.assemble_forms(geom_ord=geom_ord)
        
        if any(s>=1 for s in geom_ord):
            gradlik = df.Vector(self.pde.mpi_comm,self.pde.V.dim())
#             gradlik.axpy(-1.0,self._get_grad_misfit().vector())
            gradlik.axpy(-1.0,self._get_grad_misfit())
            if whitened:
                gradlik[:] = self.prior.C_act(gradlik,.5,op='C',transp=True)
        
        if any(s>=1.5 for s in geom_ord):
#             GNH = lambda u: self._get_metact_misfit(u).vector()
            GNH = lambda u: self._get_metact_misfit(u)
            metact = GNH
            if whitened:
                ppGNH = lambda u: self.prior.C_act(GNH(self.prior.C_act(u,.5,op='C')),.5,op='C',transp=True)
                metact = ppGNH
            
        if any(s>1 for s in geom_ord) and len(kwargs)!=0:
            # compute eigen-decomposition using randomized algorithms
            if whitened:
                # _C^(1/2) F _C^(1/2) = V D V', V' V = I
                eigs = eigen_RA(metact,dim=self.pde.V.dim(),**kwargs)
            else:
                # generalized eigen-decomposition (F, _C^(-1)), i.e. F = _C^(-1) U D U^(-1), U' _C^(-1) U = I, V = _C^(-1/2) U
                eigs = geigen_RA(metact,lambda u: self.prior.C_act(u,-1,op='K'),lambda u: self.prior.C_act(u,op='K'),dim=self.pde.V.dim(),**kwargs)
            if any(s>1.5 for s in geom_ord):
                # adjust the gradient
#                 gradlik.axpy(1.0,GNH(unknown))
                # update low-rank approximate Gaussian posterior
                self.post_Ga = Gaussian_posterior_LRapp(self.prior,eigs=eigs)
                Hu = self.prior.gen_vector()
#                 self.post_Ga.Hlr.mult(unknown.vector(), Hu)
                self.post_Ga.Hlr.mult(unknown, Hu)
                gradlik.axpy(1.0,Hu)
        
        if len(kwargs)==0:
            return loglik,gradlik,metact
        else:
            return loglik,gradlik,metact,eigs
    
    def _logpost(self,x):
        """
        Logarithm of posterior density evaluated at x.
        """
#         u=vec2fun(x,self.pde.V)
        u=self.prior.gen_vector(x)
        loglik,gradlik,_ = self.get_geom(unknown=u, geom_ord=[0,1], whitened=False)
        logpri,gradpri = self.prior.logpdf(u,grad=True)
        nlogpost = -(loglik + logpri)
        if nlogpost<0:
            print('Negative objective function value %.4f! How could it be possible?' % nlogpost)
        ngradpost = -(gradlik + gradpri).array()
        return nlogpost,ngradpost
    
    def _postHessact(self,x,a):
        """
        Posterior Hessian action evaluated at x acting on a.
        """
#         _,_,metact = self.get_geom(unknown=vec2fun(x,self.pde.V), geom_ord=[1.5], whitened=False)
        _,_,metact = self.get_geom(unknown=self.prior.gen_vector(x), geom_ord=[1.5], whitened=False)
        a = self.prior.gen_vector(a)
        GNH_post_a = (metact(a) + self.prior.C_act(a,-1,op='K')).array()
#         Fa = self.prior.gen_vector()
#         self.prior.invM.solve(Fa,metact(a))
#         GNH_post_a = (Fa + self.prior.C_act(a,-1,op='C')).array()
#         Pa = self.prior.gen_vector()
#         if type(self.prior.invK) is df.PETScLUSolver:
#             self.prior.invK.solve(Pa,a)
#         else:
#             Pa[:]=self.prior.invK.solve(a)
#         GNH_post_a = (metact(a) + Pa).array()
        return GNH_post_a
    
    def _appostHessact(self,x,a):
        """
        approximate Posterior Hessian action evaluated at x acting on a.
        """
#         _,_,_,eigs = self.get_geom(unknown=vec2fun(x,self.pde.V), geom_ord=[1.5], whitened=False, k=10)
        _,_,_,eigs = self.get_geom(unknown=self.prior.gen_vector(x), geom_ord=[1.5], whitened=False, k=10)
        self.post_Ga.eigs = eigs
#         a = self.prior.gen_vector(a)
        apGNH_post_a = self.post_Ga.postC_act(a,-1).array()
        return apGNH_post_a
    
    def get_MAP(self,SAVE=False):
        """
        Get maximum a posterior (MAP).
        """
        from scipy.optimize import minimize
        global Nfeval
        Nfeval=1
        def call_back(Xi):
            global Nfeval
            print '{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], self._logpost(Xi)[0])
            Nfeval += 1
        print '{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}'.format('Iter', ' X1', ' X2', ' X3', 'f(X)')
        res = minimize(fun=self._logpost,x0=self.prior.sample(),method='Newton-CG',jac=True,hessp=self._appostHessact,callback=call_back,options={'disp':True,'maxiter':1000})
        MAP= vec2fun(res.x,self.pde.V)
        if SAVE:
            import os
            if not hasattr(self, 'savepath'):
                cwd=os.getcwd()
                self.savepath=os.path.join(cwd,'result')
                if not os.path.exists(self.savepath):
                    print('Save path does not exist; created one.')
                    os.makedirs(self.savepath)
            MAP_file = df.HDF5File(self.pde.mpi_comm, os.path.join(self.savepath,"MAP.h5"), "w")
            MAP_file.write(MAP, "parameter")
            MAP_file.close()
        return MAP
    
    def save_soln(self,sep=False):
        """
        Save (forward/adjoint) solutions of the PDE.
        """
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
                soln = getattr(self.pde,sol_name)
            except AttributeError:
                print(self.sub_titles[i]+'solution not found!')
                pass
            else:
                if not sep:
                    df.File(os.path.join(self.savepath,sol_name+'.xml'))<<soln
                else:
                    soln = soln.split(True)
                    for j,splt in enumerate(self.titles):
                        df.File(os.path.join(self.savepath,'_'.join([splt,sol])+'.pvd'))<<soln[j]

    def _plot_vtk(self,SAVE=False):
        """
        Plotting function using VTK (default in dolfin) as backend.
        """
        for i,sol in enumerate(self.sols):
            # get solution
            try:
                soln = getattr(self.pde,'_'.join(['states',sol]))
            except AttributeError:
                print(self.sub_titles[i]+'solution not found!')
                pass
            else:
                soln = soln.split(True)
                for j,titl in enumerate(self.titles):
                    fig=df.plot(soln[j],title=self.sub_titles[i]+' '+titl,rescale=True)
                    if SAVE:
                        fig.write_png(os.path.join(self.savepath,'_'.join([titl,sol])+'.png'))

    def _plot_mpl(self,SAVE=False):
        """
        Plotting function using matplotlib as backend.
        """
        import matplotlib.pyplot as plt
        global plot
        try:
            col_bar_supp = True
            df.parameters["plotting_backend"]="matplotlib"
            plot=df.plot
        except KeyError:
            col_bar_supp = False #no colorbar support on older version
            print('Warning: plot has not been overloaded with matplotlib before version 1.7.0-dev!')
#             from util import matplot4dolfin
            if SAVE:
                savepath=self.savepath
            else:
                savepath=None
            matplot=matplot4dolfin(SAVE=SAVE,savepath=savepath)
            plot=matplot.mplot
        # codes for plotting solutions
        import matplotlib as mp
        for i,titl in enumerate(self.titles):
            fig,axes = plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True,num=i,figsize=(10,6))
            for j,ax in enumerate(axes.flat):
                # get solution
                try:
                    soln = getattr(self.pde,'_'.join(['states',self.sols[j]]))
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
#             plt.tight_layout()
            if SAVE:
                matplot.savefig(titl+'.png',bbox_inches='tight')

    def plot_soln(self,backend='matplotlib',SAVE=False):
        """
        Function to plot solutions of the PDE.
        """
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
            df.interactive()
        else:
            raise Exception(backend+'not found!')

    def test(self,SAVE=False,PLOT=False,SNR=10,chk_fd=False,h=1e-4):
        """
        Test results by the adjoint method against those by the finite difference method.
        """
        # generate theta
        unknown=self.prior.sample()
        # obtain observations
        obs,sd_noise,idx,loc=get_obs(pde4inf=self.pde,unknown=unknown,prior=self.prior,SNR=SNR)
        # re-define data misfit class
        self.misfit=data_misfit(self.pde,obs,1./sd_noise**2,idx,loc)

        import time
        # obtain the geometric quantities
        print('\n\nObtaining geometric quantities with Adjoint method...')
        start = time.time()
        loglik,gradlik,Fv = self.get_geom(unknown,[0,1,1.5])
        Funknown=Fv(unknown)
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
            v = self.prior.sample()
            ## gradient
            print('\nChecking gradient:')
            unknown_p = self.prior.gen_vector(unknown); unknown_p.axpy(h,v)
            loglik_p,_,_ = self.get_geom(unknown_p)
#             self.pde.set_forms(vec2fun(unknown_p,self.pde.V))
#             loglik_p = -self._get_misfit()
            unknown_m = self.prior.gen_vector(unknown); unknown_m.axpy(-h,v)
            loglik_m,_,_ = self.get_geom(unknown_m)
#             self.pde.set_forms(vec2fun(unknown_m,self.pde.V))
#             loglik_m = -self._get_misfit()
            dloglikv_fd = (loglik_p-loglik_m)/(2*h)
            dloglikv = gradlik.inner(v)
            rdiff_gradv = np.abs(dloglikv_fd-dloglikv)/v.norm('l2')
            print('Relative difference of gradients in a random direction between adjoint and finite difference: %.10f' % rdiff_gradv)

            # random direction
            w = self.prior.sample()
            ## metric-action
            print('\nChecking Metric-action:')
            dgradvw_fd = 0
            # obtain sensitivities
            for n,idx_n in enumerate(idx):
                misfit_n=data_misfit(self.pde,obs[n],1./sd_noise**2,idx_n,loc[[n],])
                # in direction v
                unknown_p = self.prior.gen_vector(unknown); unknown_p.axpy(h,v)
                self.pde.set_forms(vec2fun(unknown_p,self.pde.V))
                u_p,_ = self.pde.soln_fwd()
                u_p_v = misfit_n._extr_soloc(u_p)
                unknown_m = self.prior.gen_vector(unknown); unknown_m.axpy(-h,v)
                self.pde.set_forms(vec2fun(unknown_m,self.pde.V))
                u_m,_ = self.pde.soln_fwd()
                u_m_v = misfit_n._extr_soloc(u_m)
                dudunknown_v=(u_p_v-u_m_v)/(2*h)
                # in direction w
                unknown_p = self.prior.gen_vector(unknown); unknown_p.axpy(h,w)
                self.pde.set_forms(vec2fun(unknown_p,self.pde.V))
                u_p,_ = self.pde.soln_fwd()
                u_p_w = misfit_n._extr_soloc(u_p)
                unknown_m = self.prior.gen_vector(unknown); unknown_m.axpy(-h,w)
                self.pde.set_forms(vec2fun(unknown_m,self.pde.V))
                u_m,_ = self.pde.soln_fwd()
                u_m_w = misfit_n._extr_soloc(u_m)
                dudunknown_w=(u_p_w-u_m_w)/(2*h)
                # Metric (Gauss-Newton Hessian) with one observation
                dgradvw_fd += dudunknown_w*dudunknown_v
            dgradvw_fd *= self.misfit.prec
            dgradvw = w.inner(Fv(v))
            rdiff_Metvw = np.abs(dgradvw_fd-dgradvw)/v.norm('l2')/w.norm('l2')
            print('Relative difference of Metrics in two random directions between adjoint and finite difference: %.10f' % rdiff_Metvw)
            end = time.time()
            print('Time used is %.4f' % (end-start))

if __name__ == '__main__':
    np.random.seed(2017)
    elliptic = Elliptic(nx=40,ny=40,SNR=10)
    elliptic.test(SAVE=False,PLOT=False,chk_fd=True,h=1e-4)
#     MAP=elliptic.get_MAP(SAVE=True)
# #     df.plot(MAP)
# #     df.interactive()
#     matplot=matplot4dolfin(SAVE=True,savepath='./result')
#     matplot.mplot(MAP)
#     matplot.save('MAP.png',bbox_inches='tight')