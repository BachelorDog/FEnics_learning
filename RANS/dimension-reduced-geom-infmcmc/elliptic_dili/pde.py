#!/usr/bin/env python
"""
Class definition of Elliptic PDE model in the DILI paper by Cui et~al (2016).
Data come from file or are generated according to the model in a mesh finer than the one for inference.
------------------------------------------------------------
written in FEniCS 1.7.0-dev, with backward support for 1.6.0
Shiwei Lan @ U of Warwick, 2016; @ Caltech, Sept. 2016
-------------------------------
Created July 30, 2016
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2016, The EQUIP/EQUiPS projects"
__license__ = "GPL"
__version__ = "0.6"
__maintainer__ = "Shiwei Lan"
__email__ = "S.Lan@warwick.ac.uk; slan@caltech.edu; lanzithinking@outlook.com"

# import modules
import dolfin as df
import ufl
import numpy as np
import scipy.sparse as sps

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')

# Optimization options for the form compiler
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True

class _source_term(df.Expression):
    """
    Source/sink term f, defined by the superposition of four weighted Gaussian plumes with standard deviation 0.05, 
    centered at [0.3,0.3], [0.7,0.3], [0.7,0.7], [0.3,0.7], with weights {2,-3,3,-2}.
    """
    def __init__(self,mean=np.array([[0.3,0.3], [0.7,0.3], [0.7,0.7], [0.3,0.7]]),sd=0.05,wts=[2,-3,3,-2],**kwargs):
        self.mean=mean
        self.sd=sd
        self.wts=wts
    def eval(self,value,x):
        pdfs=1./np.sqrt(2*np.pi)/self.sd*np.exp(-.5*np.sum((self.mean-x)**2,axis=1)/self.sd**2)
        value[0]=pdfs.dot(self.wts)
    def plot(self,V):
        f_source=df.interpolate(self, V)
        from util import matplot4dolfin
        matplot=matplot4dolfin()
        fig=matplot.plot(f_source)
#         matplot.show()
        return fig

class ellipticPDE:
    """
    The classic elliptic PDE:
    -nabla (kappa nabla p) = f, on [0,1]^2
    <kappa nabla p, n> = 0 on boundary
    int_{boundary} p ds = 0
    """
    def __init__(self,nx=40,ny=40,nugg=1.0e-20):
        """
        Initialize the elliptic PDE with mesh size.
        """
        # 1. Define the Geometry
        # mesh size
        self.nx=nx; self.ny=ny;
        self.nugg = df.Constant(nugg)
        # set FEM
        self.set_FEM()

        # count PDE solving times
        self.soln_count = np.zeros(4)
        # 0-3: number of solving (forward,adjoint,2ndforward,2ndadjoint) equations respectively

    def set_FEM(self):
        """
        Define finite element space of elliptic PDE.
        """
        self.mesh = df.UnitSquareMesh(self.nx, self.ny)
        self.mpi_comm = self.mesh.mpi_comm()

        # boundaries
        self.boundaries = df.FacetFunction("size_t", self.mesh, 0)
        self.ds = df.ds(subdomain_data=self.boundaries)

        # 2. Define the finite element spaces and build mixed space
        try:
            V_fe = df.FiniteElement("CG", self.mesh.ufl_cell(), 2)
            L_fe = df.FiniteElement("CG", self.mesh.ufl_cell(), 2)
            self.V = df.FunctionSpace(self.mesh, V_fe)
            self.W = df.FunctionSpace(self.mesh, V_fe * L_fe)
        except TypeError:
            print('Warning: ''MixedFunctionSpace'' has been deprecated in DOLFIN version 1.7.0.')
            print('It will be removed from version 2.0.0.')
            self.V = df.FunctionSpace(self.mesh, 'CG', 2)
            L = df.FunctionSpace(self.mesh, 'CG', 2)
            self.W = self.V * L

        # 3. Define boundary conditions
        bc_lagrange = df.DirichletBC(self.W.sub(1), df.Constant(0.0), "fabs(x[0])>2.0*DOLFIN_EPS & fabs(x[0]-1.0)>2.0*DOLFIN_EPS & fabs(x[1])>2.0*DOLFIN_EPS & fabs(x[1]-1.0)>2.0*DOLFIN_EPS")

        self.ess_bc = [bc_lagrange]

        # Create adjoint boundary conditions (homogenized forward BCs)
        def homogenize(bc):
            bc_copy = df.DirichletBC(bc)
            bc_copy.homogenize()
            return bc_copy
        self.adj_bcs = [homogenize(bc) for bc in self.ess_bc]
    
    def set_forms(self,unknown,geom_ord=[0]):
        """
        Set up weak forms of elliptic PDE.
        """
        if any(s>=0 for s in geom_ord):
            ## forms for forward equation ##
            # 4. Define variational problem
            # functions
            if not hasattr(self, 'states_fwd'):
                self.states_fwd = df.Function(self.W)
            # u, l = df.split(self.states_fwd)
            u, l = df.TrialFunctions(self.W)
            v, m = df.TestFunctions(self.W)
            f = _source_term(degree=2)
            # variational forms
            if 'true' in str(type(unknown)):
                unknown = df.interpolate(unknown, self.V)
            self.F = df.exp(unknown)*df.inner(df.grad(u), df.grad(v))*df.dx + (u*m + v*l)*self.ds - f*v*df.dx + self.nugg*l*m*df.dx
#             self.dFdstates = df.derivative(self.F, self.states_fwd) # Jacobian
#             self.a = unknown*df.inner(df.grad(u), df.grad(v))*df.dx + (u*m + v*l)*self.ds + self.nugg*l*m*df.dx
#             self.L = f*v*df.dx
        if any(s>=1 for s in geom_ord):
            ## forms for adjoint equation ##
            # Set up the objective functional J
#             u,_,_ = df.split(self.states_fwd)
#             J_form = obj.form(u)
            # Compute adjoint of forward operator
            F2 = df.action(self.F, self.states_fwd)
            self.dFdstates = df.derivative(F2, self.states_fwd)    # linearized forward operator
            args = ufl.algorithms.extract_arguments(self.dFdstates) # arguments for bookkeeping
            self.adj_dFdstates = df.adjoint(self.dFdstates, reordered_arguments=args) # adjoint linearized forward operator
#             self.dJdstates = df.derivative(J_form, self.states_fwd, df.TestFunction(self.W)) # derivative of functional with respect to solution
#             self.dirac_1 = obj.ptsrc(u,1) # dirac_1 cannot be initialized here because it involves evaluation
            ## forms for gradient ##
            self.dFdunknown = df.derivative(F2, unknown)
            self.adj_dFdunknown = df.adjoint(self.dFdunknown)
        
#         if any(s>1 for s in ord):
#             ## forms for 2nd adjoint equation ##
# #             self.d2Jdstates = df.derivative(self.dJdstates, self.states_fwd) # 2nd order derivative of functional with respect to solution
#             self.dirac_2 = obj.ptsrc(ord=2) # dirac_1 cannot be initialized here because it is independent of u

            # do some assembling here to avoid repetition
#             self.assemble_forms()

    def assemble_forms(self,geom_ord=[0]):
        """
        Assemble some forms required for calculating geometric quantities.
        """
        # do some assembling here to avoid repetition
        if any(s>=1 for s in geom_ord):
            # for adjoints:
            self.adj_dFdstates_assemb = df.PETScMatrix();
            df.assemble(self.adj_dFdstates, tensor=self.adj_dFdstates_assemb)
            # for grad and metact:
            self.adj_dFdunknown_assemb = df.PETScMatrix()
            df.assemble(self.adj_dFdunknown, tensor=self.adj_dFdunknown_assemb)
        if any(s>1 for s in geom_ord):
            # for fwd2 (and fwd for nonlinear problem):
            self.dFdstates_assemb = df.PETScMatrix()
            df.assemble(self.dFdstates, tensor=self.dFdstates_assemb)
            [bc.apply(self.dFdstates_assemb) for bc in self.adj_bcs]
            self.dFdunknown_assemb = df.PETScMatrix()
            df.assemble(self.dFdunknown, tensor=self.dFdunknown_assemb)
    
    def soln_fwd(self):
        """
        Solve the forward equation.
        F = 0
        """
        # 5. Solve (non)linear variational problem
#         df.solve(self.F==0,self.states_fwd,self.ess_bc,J=self.dFdstates)
#         self.states_fwd = df.Function(self.W)
        df.solve(df.lhs(self.F)==df.rhs(self.F),self.states_fwd,self.ess_bc)
#         df.solve(self.a==self.L,self.states_fwd,self.ess_bc)
        self.soln_count[0] += 1
        u_fwd, l_fwd = df.split(self.states_fwd)
        return u_fwd, l_fwd
    
    def soln_adj(self,obj):
        """
        Solve the adjoint equation.
        < adj_dFdstates, states_adj > = dJdstates
        """
        self.states_adj = df.Function(self.W) # adjoint states
        # Solve adjoint PDE < adj_dFdstates, states_adj > = dJdstates
#         df.solve(self.adj_dFdstates == self.dJdstates , self.states_adj, self.adj_bcs)
#         A,b = df.assemble_system(self.adj_dFdstates, self.dJdstates, self.adj_bcs)
#         df.solve(A, self.states_adj.vector(), b)
#         error: assemble (solve) point integral (J) has supported underlying FunctionSpace no more than CG1; have to use PointSource? Yuk!
        
        u_fwd,_ = df.split(self.states_fwd)
#         if not df.has_petsc4py():
#             warnings.warn('Configure dolfin with petsc4py to run faster!')
#             dirac_1 = obj.ptsrc(u_fwd,ord=1)
#             rhs_adj = df.Vector(self.mpi_comm,self.W.dim())
#             [delta.apply(rhs_adj) for delta in dirac_1]
#         else:
#             rhs_adj = df.PETScVector(self.mpi_comm,self.W.dim())
#             val_dirac_1,idx_dirac_1 = obj.dirac(u_fwd,ord=1)
#             rhs_adj.vec()[idx_dirac_1] = val_dirac_1
        rhs_adj = df.Vector(self.mpi_comm,self.W.dim())
        val_dirac_1,idx_dirac_1 = obj.dirac(u_fwd,ord=1)
        rhs_adj[idx_dirac_1] = val_dirac_1
#             np.allclose(rhs_adj.array(),rhs_adj1.vec())
        
        [bc.apply(self.adj_dFdstates_assemb,rhs_adj) for bc in self.adj_bcs]

        df.solve(self.adj_dFdstates_assemb, self.states_adj.vector(), rhs_adj)
        self.soln_count[1] += 1
        u_adj, l_adj = df.split(self.states_adj)
        return u_adj, l_adj

    def soln_fwd2(self,u_actedon):
        """
        Solve the 2nd order forward equation.
        < dFdstates, states_fwd2 > = < dFdunknown, u_actedon >
        """
        self.states_fwd2 = df.Function(self.W) # 2nd forward states
        # Solve 2nd forward PDE < dFdstates, states_fwd2 > = < dFdunknown, u_actedon >
#         df.solve(self.dFdstates == df.action(self.dFdunknown, u_actedon), self.states_fwd2, self.adj_bcs) # ToDo: check the boundary for fwd2
#         A,b = df.assemble_system(self.dFdstates, df.action(self.dFdunknown, u_actedon), self.adj_bcs)
#         df.solve(A, self.states_fwd2.vector(), b)

        rhs_fwd2 = df.PETScVector()
#         df.assemble(df.action(self.dFdunknown, u_actedon), tensor=rhs_fwd2)
        self.dFdunknown_assemb.mult(u_actedon.vector(),rhs_fwd2)

        [bc.apply(rhs_fwd2) for bc in self.adj_bcs]

        df.solve(self.dFdstates_assemb, self.states_fwd2.vector(), rhs_fwd2)
        self.soln_count[2] += 1
        u_fwd2, l_fwd2 = df.split(self.states_fwd2)
        return u_fwd2, l_fwd2

    def soln_adj2(self,obj,**kwargs):
        """
        Solve the 2nd order adjoint equation.
        < adj_dFdstates, states_adj2 > = < d2Jdstates, states_fwd2 >
        """
        self.states_adj2 = df.Function(self.W) # 2nd forward states
        # Solve 2nd adjoint PDE < adj_dFdstates, states_adj2 > = < d2Jdstates, states_fwd2 >
#         df.solve(self.adj_dFdstates == df.action(self.d2Jdstates, self.states_fwd2), self.states_adj2, self.adj_bcs)
#         A,b = df.assemble_system(self.adj_dFdstates, df.action(self.d2Jdstates, self.states_fwd2), self.adj_bcs)
#         df.solve(A, self.states_adj2.vector(), b)

#         rhs_adj2 = df.PETScVector()
#         df.assemble(df.action(self.d2Jdstates, self.states_fwd2), tensor=rhs_adj2)

#         u_fwd2,_ = df.split(self.states_fwd2)
        u_fwd2 = kwargs.pop('u_fwd2',df.split(self.states_fwd2)[0]) # generalized for root metric
#         if not df.has_petsc4py():
#             warnings.warn('Configure dolfin with petsc4py to run faster!')
#             dirac_2 = obj.ptsrc(u_fwd2,ord=2)
#             rhs_adj2 = df.Vector(self.mpi_comm,self.W.dim())
#             [delta.apply(rhs_adj2) for delta in dirac_2]
#         else:
#             rhs_adj2 = df.PETScVector(self.mpi_comm,self.W.dim())
#             val_dirac_2,idx_dirac_2 = obj.dirac(u_fwd2,ord=2)
#             rhs_adj2.vec()[idx_dirac_2] = val_dirac_2
        rhs_adj2 = df.Vector(self.mpi_comm,self.W.dim())
        val_dirac_2,idx_dirac_2 = obj.dirac(u_fwd2,ord=2)
        rhs_adj2[idx_dirac_2] = val_dirac_2
#             np.allclose(rhs_adj2.array(),rhs_adj12.vec())

        [bc.apply(rhs_adj2) for bc in self.adj_bcs]

        df.solve(self.adj_dFdstates_assemb, self.states_adj2.vector(), rhs_adj2)
        self.soln_count[3] += 1
        u_adj2, l_adj2 = df.split(self.states_adj2)
        return u_adj2, l_adj2
    
    def plot_soln(self,soln_f):
        """
        Plot solution function.
        """
        from util import matplot4dolfin
        matplot=matplot4dolfin()
        fig=matplot.plot(soln_f)
#         matplot.show()
        return fig
    
if __name__ == '__main__':
    np.random.seed(2017)
    # define PDE
    pde=ellipticPDE()
    # get the forcing function
    force = _source_term(degree=2)
    # set the true parameter function
    from misfit import _true_coeff
    truth = _true_coeff(degree=0)
    pde.set_forms(unknown=truth)
    # solve the forward equation with the forcing and the true parameter
    _,_=pde.soln_fwd()
    u_fwd,_=pde.states_fwd.split(True)
    # plot
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    fig,axes = plt.subplots(nrows=1,ncols=2,sharex=True,figsize=(14,5))
    sub_figs = [None]*len(axes)
    # plot the truth
    plt.axes(axes[0])
    sub_figs[0]=force.plot(pde.V)
    plt.title('The forcing term',fontsize=12)
    plt.colorbar(sub_figs[0])
#     ax.set_title('True transmissivity field',fontsize=12)
    # plot observations
    plt.axes(axes[1])
    sub_figs[1]=pde.plot_soln(u_fwd)
    plt.title('Solution of the potential',fontsize=12)
    plt.colorbar(sub_figs[1])
#     ax.set_title('Observations on selected locations',fontsize=12)
#     cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
#     plt.colorbar(sub_fig, cax=cax, **kw)
#     from util.common_colorbar import common_colorbar
#     fig=common_colorbar(fig,axes,sub_figs)
    # fig.tight_layout()
    plt.savefig('./result/force_soln.png',bbox_inches='tight')
    plt.show()
    