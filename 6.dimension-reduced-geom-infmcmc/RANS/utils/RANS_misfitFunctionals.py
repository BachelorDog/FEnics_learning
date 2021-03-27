import dolfin as dl
import numpy as np
from hippylib import *
from RANS_loadDNSdata import VelocityDNS, KDNS

class DNSDataMisfit(Misfit):
    """
    Class that models the misfit componenet of the cost functional.
    In the following x will denote the variable [u, m, p], denoting respectively
    the state u, the parameter m, and the adjoint variable p.
    
    The methods in the class misfit will usually access the state u and possibly the
    parameter m. The adjoint variables will never be accessed. 
    """
    
    def __init__(self, Vh_STATE, Vhs, bcs0, datafile, dx = dl.dx):
        self.dx = dx
        x, y, U, V, uu, vv, ww, uv, k = np.loadtxt(datafile,skiprows=2, unpack=True)
        u_fun_mean = VelocityDNS(x=x, y=y, U=U, V=V, symmetrize = True, coflow=0.)
        u_fun_data = VelocityDNS(x=x, y=y, U=U, V=V, symmetrize = False, coflow=0.)
        k_fun_mean = KDNS(x=x, y=y, k=k, symmetrize = True)
        k_fun_data = KDNS(x=x, y=y, k=k, symmetrize = False)
        
        u_data = dl.interpolate(u_fun_data, Vhs[0])
        k_data = dl.interpolate(k_fun_data, Vhs[2])
        
        noise_var_u = dl.assemble(dl.inner(u_data-u_fun_mean, u_data-u_fun_mean)*self.dx)
        noise_var_k = dl.assemble(dl.inner(k_data-k_fun_mean, k_data-k_fun_mean)*self.dx)
        
        u_trial, p_trial, k_trial, e_trial = dl.TrialFunctions(Vh_STATE)
        u_test,  p_test,  k_test,  e_test  = dl.TestFunctions(Vh_STATE)
        
        Wform = dl.Constant(1./noise_var_u)*dl.inner(u_trial, u_test)*self.dx + \
                dl.Constant(1./noise_var_k)*dl.inner(k_trial, k_test)*self.dx
               
        self.W = dl.assemble(Wform)
        dummy = dl.Vector()
        self.W.init_vector(dummy,0)
        [bc.zero(self.W) for bc in bcs0]
        Wt = Transpose(self.W)
        [bc.zero(Wt) for bc in bcs0]
        self.W = Transpose(Wt)
        
        xfun = dl.Function(Vh_STATE)
        assigner = dl.FunctionAssigner(Vh_STATE, Vhs)
        assigner.assign(xfun, [u_data, dl.Function(Vhs[1]), k_data, dl.Function(Vhs[3])])
        self.d = xfun.vector()
        
    
    def cost(self,x):
        """Given x evaluate the cost functional.
        Only the state u and (possibly) the parameter m are accessed. """
        diff = x[STATE] - self.d
        return .5*diff.inner(self.W*diff)
        
    def grad(self, i, x, out):
        """Given the state and the paramter in x, compute the partial gradient of the misfit
        functional in with respect to the state (i == STATE) or with respect to the parameter (i == PARAMETER).
        """
        if i == STATE:
            self.W.mult(x[STATE]-self.d, out)
        elif i == PARAMETER:
            out.zero()
        else:
            raise IndexError()
                
    def setLinearizationPoint(self,x):
        """Set the point for linearization."""
        pass
        
    def apply_ij(self,i,j, dir, out):
        """Apply the second variation \delta_ij (i,j = STATE,PARAMETER) of the cost in direction dir."""
        if i == STATE and j == STATE:
            self.W.mult(dir, out)
        else:
            out.zero()
            
class DNSVelocityMisfit(Misfit):
    """
    Class that models the misfit componenet of the cost functional.
    In the following x will denote the variable [u, m, p], denoting respectively
    the state u, the parameter m, and the adjoint variable p.
    
    The methods in the class misfit will usually access the state u and possibly the
    parameter m. The adjoint variables will never be accessed. 
    """
    
    def __init__(self, Vh_STATE, Vhs, bcs0, datafile, dx = dl.dx):
        self.dx = dx
        x, y, U, V, uu, vv, ww, uv, k = np.loadtxt(datafile,skiprows=2, unpack=True)
        u_fun_mean = VelocityDNS(x=x, y=y, U=U, V=V, symmetrize = True, coflow=0.)
        u_fun_data = VelocityDNS(x=x, y=y, U=U, V=V, symmetrize = False, coflow=0.)
        
        u_mean = dl.interpolate(u_fun_mean, Vhs[0])
        u_data = dl.interpolate(u_fun_data, Vhs[0])

        q_order = dl.parameters["form_compiler"]["quadrature_degree"]
        dl.parameters["form_compiler"]["quadrature_degree"] = 6
        noise_var_u = dl.assemble(dl.inner(u_fun_data-u_mean, u_fun_data-u_mean)*self.dx,
                                  form_compiler_parameters = dl.parameters["form_compiler"])
        dl.parameters["form_compiler"]["quadrature_degree"] = q_order
        
        noise_var_u = 1.e-3
        mpi_comm = Vh_STATE.mesh().mpi_comm()
        rank = dl.MPI.rank(mpi_comm)
        if rank == 0:
            print "Noise Variance = {0}".format(noise_var_u)
        
        if Vh_STATE.num_sub_spaces() == 2:
            u_trial, p_trial = dl.TrialFunctions(Vh_STATE)
            u_test,  p_test  = dl.TestFunctions(Vh_STATE)
        elif Vh_STATE.num_sub_spaces() == 3:
            u_trial, p_trial, g_trial = dl.TrialFunctions(Vh_STATE)
            u_test,  p_test,  g_test  = dl.TestFunctions(Vh_STATE)
        else:
            raise InputError()
        
        Wform = dl.Constant(1./noise_var_u)*dl.inner(u_trial, u_test)*self.dx
               
        self.W = dl.assemble(Wform)
        dummy = dl.Vector()
        self.W.init_vector(dummy,0)
        [bc.zero(self.W) for bc in bcs0]
        Wt = Transpose(self.W)
        [bc.zero(Wt) for bc in bcs0]
        self.W = Transpose(Wt)
        
        xfun = dl.Function(Vh_STATE)
        assigner = dl.FunctionAssigner(Vh_STATE, Vhs)
        if Vh_STATE.num_sub_spaces() == 2:
            assigner.assign(xfun, [u_data, dl.Function(Vhs[1])])
        elif Vh_STATE.num_sub_spaces() == 3:
            assigner.assign(xfun, [u_data, dl.Function(Vhs[1]), dl.Function(Vhs[2])])
            
        self.d = xfun.vector()
        
    
    def cost(self,x):
        """Given x evaluate the cost functional.
        Only the state u and (possibly) the parameter m are accessed. """
        diff = x[STATE] - self.d
        return .5*diff.inner(self.W*diff)
    
    def grad(self, i, x, out):
        """Given the state and the paramter in x, compute the partial gradient of the misfit
        functional in with respect to the state (i == STATE) or with respect to the parameter (i == PARAMETER).
        """
        if i == STATE:
            self.W.mult(x[STATE]-self.d, out)
        elif i == PARAMETER:
            out.zero()
        else:
            raise IndexError()
        
    def setLinearizationPoint(self,x):
        """Set the point for linearization."""
        pass
        
    def apply_ij(self,i,j, dir, out):
        """Apply the second variation \delta_ij (i,j = STATE,PARAMETER) of the cost in direction dir."""
        if i == STATE and j == STATE:
            self.W.mult(dir, out)
        else:
            out.zero()
            
class AlgModelMisfit(Misfit):
    """
    Class that models the misfit componenet of the cost functional.
    In the following x will denote the variable [u, m, p], denoting respectively
    the state u, the parameter m, and the adjoint variable p.
    
    The methods in the class misfit will usually access the state u and possibly the
    parameter m. The adjoint variables will never be accessed. 
    """
    
    def __init__(self, Vh_STATE, Vhs, geo, bcs0, datafile, variance_u, variance_g):
        if hasattr(geo, "dx"):
            self.dx = geo.dx(geo.PHYSICAL)
        else:
            self.dx = dl.dx
            
        self.ds = geo.ds(geo.AXIS)
        
        x, y, U, V, uu, vv, ww, uv, k = np.loadtxt(datafile,skiprows=2, unpack=True)
        u_fun_data = VelocityDNS(x=x, y=y, U=U, V=V, symmetrize = True, coflow=0.)
        
        u_data = dl.interpolate(u_fun_data, Vhs[0])
                
        if Vh_STATE.num_sub_spaces() == 3:
            u_trial, p_trial, g_trial = dl.TrialFunctions(Vh_STATE)
            u_test,  p_test,  g_test  = dl.TestFunctions(Vh_STATE)
        else:
            raise InputError()
        
        Wform = dl.Constant(1./variance_u)*dl.inner(u_trial, u_test)*self.dx +\
                dl.Constant(1./variance_g)*g_trial*g_test*self.ds
               
        self.W = dl.assemble(Wform)
        dummy = dl.Vector()
        self.W.init_vector(dummy,0)
        [bc.zero(self.W) for bc in bcs0]
        Wt = Transpose(self.W)
        [bc.zero(Wt) for bc in bcs0]
        self.W = Transpose(Wt)
        
        xfun = dl.Function(Vh_STATE)
        assigner = dl.FunctionAssigner(Vh_STATE, Vhs)
        assigner.assign(xfun, [u_data, dl.Function(Vhs[1]), dl.interpolate(dl.Constant(1.), Vhs[2])])
            
        self.d = xfun.vector()
        
    
    def cost(self,x):
        """Given x evaluate the cost functional.
        Only the state u and (possibly) the parameter m are accessed. """
        diff = x[STATE] - self.d
        return .5*diff.inner(self.W*diff)
        
    def grad(self, i, x, out):
        """Given the state and the paramter in x, compute the partial gradient of the misfit
        functional in with respect to the state (i == STATE) or with respect to the parameter (i == PARAMETER).
        """
        if i == STATE:
            self.W.mult(x[STATE]-self.d, out)
        elif i == PARAMETER:
            out.zero()
        else:
            raise IndexError()    

    
    def setLinearizationPoint(self,x):
        """Set the point for linearization."""
        pass
        
    def apply_ij(self,i,j, dir, out):
        """Apply the second variation \delta_ij (i,j = STATE,PARAMETER) of the cost in direction dir."""
        if i == STATE and j == STATE:
            self.W.mult(dir, out)
        else:
            out.zero()
            
class AlgModelMisfitFSE(Misfit):
    """
    Class that models the misfit componenet of the cost functional.
    In the following x will denote the variable [u, m, p], denoting respectively
    the state u, the parameter m, and the adjoint variable p.
    
    The methods in the class misfit will usually access the state u and possibly the
    parameter m. The adjoint variables will never be accessed. 
    """
    
    def __init__(self, Vh_STATE, Vhs, weights, geo, bcs0, datafile, variance_u, variance_g):
        if hasattr(geo, "dx"):
            self.dx = geo.dx(geo.PHYSICAL)
        else:
            self.dx = dl.dx
            
        self.ds = geo.ds(geo.AXIS)
        
        x, y, U, V, uu, vv, ww, uv, k = np.loadtxt(datafile,skiprows=2, unpack=True)
        u_fun_data = VelocityDNS(x=x, y=y, U=U, V=V, symmetrize = True, coflow=0.)
        
        u_data = dl.interpolate(u_fun_data, Vhs[0])
                
        if Vh_STATE.num_sub_spaces() == 3:
            u_trial, p_trial, g_trial = dl.TrialFunctions(Vh_STATE)
            u_test,  p_test,  g_test  = dl.TestFunctions(Vh_STATE)
        else:
            raise InputError()
        
        Wform = dl.Constant(1./variance_u)*dl.inner(u_trial, u_test)*self.dx +\
                dl.Constant(1./variance_g)*g_trial*g_test*self.ds
               
        self.W = dl.assemble(Wform)
        dummy = dl.Vector()
        self.W.init_vector(dummy,0)
        [bc.zero(self.W) for bc in bcs0]
        Wt = Transpose(self.W)
        [bc.zero(Wt) for bc in bcs0]
        self.W = Transpose(Wt)
        
        xfun = dl.Function(Vh_STATE)
        assigner = dl.FunctionAssigner(Vh_STATE, Vhs)
        assigner.assign(xfun, [u_data, dl.Function(Vhs[1]), dl.interpolate(dl.Constant(1.), Vhs[2])])
            
        self.d = xfun.vector()
        
        self.w = (weights * 0.5)
        
    def E_stateQuad(self, state):
        assert state.n_incr == self.w.shape[0]
        E_x = state.s0.copy()
        for ii in range(state.n_incr):
            E_x.axpy(self.w[ii], state.s2[ii])
        return E_x
        
    
    def cost(self,x):
        """Given x evaluate the cost functional.
        Only the state u and (possibly) the parameter m are accessed. """
        E_x = self.E_stateQuad(x[STATE])
        diff = E_x - self.d
        return .5*diff.inner(self.W*diff)
        
    
    def grad(self,i,x,rhs):
        """Given the state and the paramter in x, compute the partial gradient of the misfit
        functional in with respect to the state (i == STATE) or with respect to the parameter (i == PARAMETER).
        """
        if i == STATE:
            assert rhs.n_incr == self.w.shape[0]
            E_x = self.E_stateQuad(x[STATE])
            mdiff = E_x - self.d
            rhs.zero()
            self.W.mult(mdiff, rhs.s0)
            for ii in range(rhs.n_incr):
                rhs.s2[ii].axpy(self.w[ii], rhs.s0)
        elif i == PARAMETER:
            rhs.zero()
        else:
            raise IndexError()
          
    def setLinearizationPoint(self,x):
        """Set the point for linearization."""
        pass
        
    def apply_ij(self,i,j, dir, out):
        """Apply the second variation \delta_ij (i,j = STATE,PARAMETER) of the cost in direction dir."""
        if i == STATE and j == STATE:
            assert dir.n_incr == self.w.shape[0]
            assert out.n_incr == self.w.shape[0]
            out.zero()
            self.W.mult(dir.s0, out.s0)
            for ii in range(out.n_incr):
                self.W.mult(dir.s2[ii], out.s2[ii])
                out.s0.axpy(self.w[ii], out.s2[ii])
                out.s2[ii].zero()
            for ii in range(out.n_incr):
                out.s2[ii].axpy(self.w[ii], out.s0)
        else:
            out.zero()
    
    