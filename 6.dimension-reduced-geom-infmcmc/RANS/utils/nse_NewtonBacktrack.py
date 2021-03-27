'''
Created on Jan 26, 2016

@author: uvilla
'''

import dolfin as dl
import numpy as np
import math

class NewtonBacktrack:
    """
    Newton-type non linear solver with backtracking line-search.
    Parameters for the nonlinear solver are set through the attribute parameter.
    - "rel_tolerance": the relative tolerance (1e-6 default)
    - "abs_tolerance": the absolute tolerance (1e-12 default)
    - "max_iter": the maximum number of iterations (200 default)
    - "print_level": controls verbosity of the solver to screen
                     -1: no output to screen
                      0: final residual, and convergence reason
                      1: print current residual and alpha at each iteration
    - "max_backtracking": maximum number of backtracking iterations
    """
    def __init__(self, problem, Wr=None, callback=None, extra_args=()):
        """
        Constructor
        
        INPUT:
        - problem: an object of type NonlinearProblem. It provides the residual and the Jacobian
        - Wr: an s.p.d. operator used to compute a weighted norm.
            This object must provide the methods:
            Wr.norm(r):= sqrt( r, Wr*r) to compute the weighted norm of r
            Wr.inner(r1,r2) := (r1, Wr*r2) to compute the weighted inner product of r1 and r2
            If Wr is None the standard l2-norm is used
        - callback: a function handler to perform additional postprocessing (such as output to paraview)
                    of the current solution
        - extra_args: a tuple of additional parameters to evaluate the residual and Jacobian
        """   
        self.parameters = {}
        self.parameters["rel_tolerance"]         = 1e-6
        self.parameters["abs_tolerance"]         = 1e-12
        self.parameters["max_iter"]              = 200
        self.parameters["print_level"]           = 0
        self.parameters["max_backtracking"]      = 20
        
        self.problem = problem
        self.Wr = Wr
                
        self.callback = callback
        self.extra_args = extra_args
        
        self.final_it = 0
        self.final_norm = 0
        self.initial_norm = 0
        self.converged = False
                            
    def _norm_r(self,r):
        if self.Wr is None:
            return r.norm("l2")
        else:
            return self.Wr.norm(r)
    
    def _inner_r(self,r1,r2):
        if self.Wr is None:
            return r1.inner(r2)
        else:
            return self.Wr.inner(r1,r2)
        
    def solve(self, x):
        
        rtol = self.parameters["rel_tolerance"]
        atol = self.parameters["abs_tolerance"]
        max_iter = self.parameters["max_iter"]
        max_backtracking = self.parameters["max_backtracking"]
        
        self.problem.applyBC(x)
        
        if self.callback is not None:
            self.callback(x)
        
        r = self.problem.residual(x, self.extra_args)
        
        norm_r0 = self._norm_r(r)
        norm_r = norm_r0
        
        self.initial_norm = norm_r0
        
        tol = max(atol, rtol*norm_r0)
        
        self.converged = False
        it = 0
                
        d = dl.Vector()
                
        if(self.parameters["print_level"] >= 1):
                print "\n{0:3} {1:15} {2:15}".format(
                      "It", "||r||", "alpha")
                
        if self.parameters["print_level"] >= 1:
                print "{0:3d} {1:15e} {2:15e}".format(
                        it, norm_r0, 1.)
        
        while it < max_iter and not self.converged:
            
            J = self.problem.Jacobian(x, self.extra_args)
            
            JSolver = dl.PETScLUSolver()
            JSolver.set_operator(J)
            JSolver.solve(d, -r)

            
            alpha = 1.
            backtracking_converged = False
            j = 0
            while j < max_backtracking and not backtracking_converged:
                x_new = x + alpha*d
                r = self.problem.residual(x_new, self.extra_args)
                try:
                    norm_r_new = self._norm_r(r)
                except:
                    norm_r_new = norm_r+1.
                if norm_r_new  <= norm_r:
                    x.axpy(alpha, d)
                    norm_r = norm_r_new
                    backtracking_converged = True
                else:
                    alpha = .5*alpha
                    j = j+1
                                        
            if not backtracking_converged:
                if self.parameters["print_level"] >= 0:
                    print "Backtracking failed at iteration", it, ". Residual norm is ", norm_r
                self.converged = False
                self.final_it = it
                self.final_norm = norm_r
                break
                
                
            if norm_r_new < tol:
                if self.parameters["print_level"] >= 0:
                    print "Converged in ", it, "iterations with final residual norm", norm_r_new
                self.final_norm = norm_r_new
                self.converged = True
                self.final_it = it
                break
                            
            it = it+1
                                
            if self.parameters["print_level"] >= 1:
                print "{0:3d} {1:15e} {2:15e}".format(
                        it,  norm_r, alpha)
    
            if self.callback is not None:
                self.callback(x)
                
        if not self.converged:
            self.final_norm = norm_r_new
            self.final_it = it
            if self.parameters["print_level"] >= 0:
                    print "Not Converged in ", it, "iterations. Final residual norm", norm_r_new
               
        
        
        