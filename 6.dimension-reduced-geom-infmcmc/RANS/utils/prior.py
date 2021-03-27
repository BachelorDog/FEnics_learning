import dolfin as dl
import numpy as np
from hippylib import _Prior

class FiniteDimensionalGaussianR:
    def __init__(self, R):
        self.R = R
        
    def init_vector(self,x, dim):
        self.R.init_vector(x,1)
        
    def inner(self,x,y):
        Rx = dl.Vector()
        self.init_vector(Rx,0)
        self.mult(x, Rx)
        return Rx.inner(y)
        
    def mult(self,x,y):
        self.R.mult(x, y)
    

class FiniteDimensionalGaussianPrior(_Prior):
    def __init__(self, Vh, sigma2, mean=None, max_iter=100, rel_tol = 1e-9):
        
        self.sigma2 = sigma2
            
        trial = dl.TrialFunction(Vh)
        test  = dl.TestFunction(Vh)
            
        varfM = dl.inner(trial, test)*dl.dx
            
        self.M = dl.assemble(varfM)
        self.M.zero()
        self.M.ident_zeros()
        
        self.Rm = dl.assemble(varfM)
        self.Rm.zero()
        sigma2inv_vector = dl.Vector()
        self.Rm.init_vector(sigma2inv_vector, 0)
        sigma2inv_vector[:] = np.power(sigma2, -1)
        self.Rm.set_diagonal(sigma2inv_vector)
        
        self.Ai = dl.assemble(varfM)
        self.Ai.zero()
        sigma_vector = dl.Vector()
        self.Ai.init_vector(sigma_vector, 0)
        sigma_vector[:] = np.sqrt(sigma2)
        self.Ai.set_diagonal(sigma_vector)

            
        self.Msolver = dl.PETScKrylovSolver("cg", "jacobi")
        self.Msolver.set_operator(self.M)
        self.Msolver.parameters["maximum_iterations"] = max_iter
        self.Msolver.parameters["relative_tolerance"] = rel_tol
        self.Msolver.parameters["error_on_nonconvergence"] = True
        self.Msolver.parameters["nonzero_initial_guess"] = False
            
        self.Rsolver = dl.PETScKrylovSolver("cg", "jacobi")
        self.Rsolver.set_operator(self.Rm)
        self.Rsolver.parameters["maximum_iterations"] = max_iter
        self.Rsolver.parameters["relative_tolerance"] = rel_tol
        self.Rsolver.parameters["error_on_nonconvergence"] = True
        self.Rsolver.parameters["nonzero_initial_guess"] = False
        
        if mean is None:
            self.mean = dl.Vector()
            self.init_vector(self.mean,0)
        else:
            self.mean = mean.copy()
            
        self.R = FiniteDimensionalGaussianR(self.Rm)
            
    def init_vector(self,x,dim):
        """
        Inizialize a vector x to be compatible with the range/domain of R.
        If dim == "noise" inizialize x to be compatible with the size of
        white noise used for sampling.
        """
        if dim == "noise":
            self.Rm.init_vector(x,1)
        else:
            self.Rm.init_vector(x,dim)
            
    def sample(self, noise, s, add_mean=True):
        """
        Given a noise ~ N(0, I) compute a sample s from the prior.
        If add_mean=True add the prior mean value to s.
        """
        self.Ai.mult(noise, s)
        
        if add_mean:
            s.axpy(1., self.mean)