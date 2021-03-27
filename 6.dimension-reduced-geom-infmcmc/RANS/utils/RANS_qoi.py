import dolfin as dl
import numpy as np
from hippylib import *


class JetThicknessQOI:
    """
    Quantity of interest for the thickness L of a turbulent jet at a given distance x0-x_inlet
    from the inlet boundary.
    
    More specifically,
    L(x0) = \int_Gamma_0 ux dy / u_cl(x0)
    
    where
    - Gamma_0 is the vertical cross-section at distance x0-x_inlet from the inlet boundary
    - ux denotes the horizontal component of the velocity field
    - u_cl is the centerline velocity (y = 0)
    """
    def __init__(self, mesh, Vh_STATE, x0):
        """
        Constructor.
        INPUTS:
        
        - mesh: the mesh
        - Vh_STATE: the finite element space for the state variable
        - x0: location at which we want to compute the jet-thickness
        """

        Vh_help = dl.FunctionSpace(mesh, "CG", 1)
        xfun = dl.interpolate(dl.Expression("x[0]", degree=1), Vh_help)
        x_coord = xfun.vector().gather_on_zero()
        
        mpi_comm = mesh.mpi_comm()
        rank = dl.MPI.rank(mpi_comm)
        nproc = dl.MPI.size(mpi_comm)
        
        # round x0 so that it is aligned with the mesh
        if nproc > 1:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            if rank == 0:
                idx = (np.abs(x_coord-x0)).argmin()
                self.x0 = x_coord[idx]
            else:
                self.x0 = None
                
            self.x0 = comm.bcast(self.x0, root=0)
        else:
            idx = (np.abs(x_coord-x0)).argmin()
            self.x0 = x_coord[idx]
            
        line_segment = dl.AutoSubDomain(lambda x: dl.near(x[0], self.x0))
        markers_f = dl.FacetFunction("size_t", mesh)
        markers_f.set_all(0)
        line_segment.mark(markers_f, 1)
        dS = dl.dS[markers_f]
        
        x_test = dl.TestFunctions(Vh_STATE)
        u_test = x_test[0]
        
        e1 = dl.Constant(("1.", "0."))
        
        self.int_u = dl.assemble(dl.avg( dl.dot(u_test,e1) )*dS(1) )
        #self.u_cl = dl.assemble( dl.dot(u_test,e1)*dP(1) )
        
        self.u_cl = dl.Function(Vh_STATE).vector()
        ps = dl.PointSource(Vh_STATE.sub(0).sub(0), dl.Point(self.x0, 0.), 1.)
        ps.apply(self.u_cl)
        
        scaling = self.u_cl.sum()
        if np.abs(scaling - 1.) > 1e-6:
            print scaling
            raise ValueError()
        
        self.state = dl.Function(Vh_STATE).vector()
        self.help = dl.Function(Vh_STATE).vector()
            
    def eval(self, x):
        """
        Evaluate the quantity of interest at a given point in the state and
        parameter space.
        
        INPUTS:
        - x = [u,m,p] is a list of the state u, parameter m, and adjoint variable p
                      at which evaluate the qoi.
                      NOTE: p can be omitted since it is not addressed.
        """
        a = self.int_u.inner(x[STATE])
        b = self.u_cl.inner(x[STATE])
        return a/b
    
    def grad(self,i, x, g):
        if i == STATE:
            self.grad_state(x,g)
        elif i==PARAMETER:
            g.zero()
        else:
            raise i
        
    def grad_state(self,x,g):
        """
        The partial derivative of the qoi with respect to the state variable.
        
        INPUTS:
        - x = [u,m,p] is a list of the state u, parameter m, and adjoint variable p at which evaluate
              the gradient w.r.t. the state. NOTE: p can be omitted since it is not addressed.
        - g: FEniCS vector to store the gradient w.r.t. the state. 
        """
        ### f = a/b
        ### df/dstate = (a'*b - a*b')/b^2
        a = self.int_u.inner(x[STATE])
        b = self.u_cl.inner(x[STATE])
        g.zero()
        g.axpy(b, self.int_u)
        g.axpy(-a, self.u_cl)
        g *= 1./(b*b)
        
    def apply_ij(self,i,j, dir, out):
        """
        Apply the second variation \delta_ij (i,j = STATE,PARAMETER) of the q.o.i. in direction dir.
        
        INPUTS:
        - i,j integer (STATE=0, PARAMETER=1) which indicates with respect to which variables differentiate
        - dir the direction in which to apply the second variation
        - out: FEniCS vector to store the second variation in the direction dir.
        
        NOTE: setLinearizationPoint must be called before calling this method.
        """
        out.zero()
        if i == STATE and j == STATE:
            ### -2*(a'*b - a*b')*b'/b^3
            a = self.int_u.inner(self.state)
            b = self.u_cl.inner(self.state)
            b_prime_dir = self.u_cl.inner(dir)
            self.help.zero()
            self.help.axpy(b, self.int_u)
            self.help.axpy(-a, self.u_cl)
            out.axpy(-b_prime_dir/(b*b*b), self.help)
            help_dir = self.help.inner(dir)
            out.axpy(-help_dir/(b*b*b), self.u_cl)
            
        
    def setLinearizationPoint(self, x):
        """
        Specify the linearization point for computation of the second variations in method apply_ij.
        
        INPUTS:
        - x = [u,m,p] is a list of the state u, parameter m, and adjoint variable p
        """
        self.state.zero()
        self.state.axpy(1., x[STATE])
        
        
class JetThicknessQOIFSE:
    """
    Quantity of interest for the thickness L of a turbulent jet at a given distance x0-x_inlet
    from the inlet boundary.
    
    More specifically,
    L(x0) = \int_Gamma_0 ux dy / u_cl(x0)
    
    where
    - Gamma_0 is the vertical cross-section at distance x0-x_inlet from the inlet boundary
    - ux denotes the horizontal component of the velocity field
    - u_cl is the centerline velocity (y = 0)
    """
    def __init__(self, mesh, Vh_STATE, x0):
        """
        Constructor.
        INPUTS:
        
        - mesh: the mesh
        - Vh_STATE: the finite element space for the state variable
        - x0: location at which we want to compute the jet-thickness
        """

        Vh_help = dl.FunctionSpace(mesh, "CG", 1)
        xfun = dl.interpolate(dl.Expression("x[0]", degree=1), Vh_help)
        x_coord = xfun.vector().gather_on_zero()
        
        mpi_comm = mesh.mpi_comm()
        rank = dl.MPI.rank(mpi_comm)
        nproc = dl.MPI.size(mpi_comm)
        
        # round x0 so that it is aligned with the mesh
        if nproc > 1:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            if rank == 0:
                idx = (np.abs(x_coord-x0)).argmin()
                self.x0 = x_coord[idx]
            else:
                self.x0 = None
                
            self.x0 = comm.bcast(self.x0, root=0)
        else:
            idx = (np.abs(x_coord-x0)).argmin()
            self.x0 = x_coord[idx]
        
        
        line_segment = dl.AutoSubDomain(lambda x: dl.near(x[0], self.x0))
        markers_f = dl.FacetFunction("size_t", mesh)
        markers_f.set_all(0)
        line_segment.mark(markers_f, 1)
        dS = dl.dS[markers_f]
        
        x_test = dl.TestFunctions(Vh_STATE)
        u_test = x_test[0]
        
        e1 = dl.Constant(("1.", "0."))
        
        self.int_u = dl.assemble(dl.avg( dl.dot(u_test,e1) )*dS(1) )
        #self.u_cl = dl.assemble( dl.dot(u_test,e1)*dP(1) )
        
        self.u_cl = dl.Function(Vh_STATE).vector()
        ps = dl.PointSource(Vh_STATE.sub(0).sub(0), dl.Point(self.x0, 0.), 1.)
        ps.apply(self.u_cl)
        
        self.state = dl.Function(Vh_STATE).vector()
        self.help = dl.Function(Vh_STATE).vector()
            
    def eval(self, x):
        """
        Evaluate the quantity of interest at a given point in the state and
        parameter space.
        
        INPUTS:
        - x = [u,m,p] is a list of the state u, parameter m, and adjoint variable p
                      at which evaluate the qoi.
                      NOTE: p can be omitted since it is not addressed.
        """
        q = 0.
        s = x[STATE].s0.copy()
        for ii in range(x[STATE].n_incr):
            s.zero()
            s.axpy(1., x[STATE].s0)
            s.axpy(1., x[STATE].s1[ii])
            s.axpy(.5, x[STATE].s1[ii])
            a = self.int_u.inner(s)
            b = self.u_cl.inner(s)
            q += a/b
        return q/float(x[STATE].n_incr)
