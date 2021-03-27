# Copyright (c) 2016, The University of Texas at Austin & University of
# California, Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 3.0 dated June 2007.


from variables import PARAMETER

class SteepestDescent:
    
    """
    Prior-preconditioned Steepest Descent to solve constrained optimization problems in the reduced parameter space.
    Globalization is performed using the armijo sufficient reduction condition (backtracking).
    The stopping criterion is based on a control on the norm of the gradient.
       
    The user must provide a model that describes the forward problem, cost functionals, and all the
    derivatives for the gradient.
    
    More specifically the model object should implement following methods:
       - generate_vector() -> generate the object containing state, parameter, adjoint
       - cost(x) -> evaluate the cost functional, report regularization part and misfit separately
       - solveFwd(out, x,tol) -> solve the possibly non linear Fwd Problem up a tolerance tol
       - solveAdj(out, x,tol) -> solve the linear adj problem
       - evalGradientParameter(x, out) -> evaluate the gradient of the parameter and compute its norm
       - Rsolver()          --> A solver for the regularization term
       
    Type help(Model) for additional information
    """
    termination_reasons = [
                           "Maximum number of Iteration reached",      #0
                           "Norm of the gradient less than tolerance", #1
                           "Maximum number of backtracking reached",   #2
                           ]
    
    def __init__(self, model):
        """
        Initialize the Steepest Descent with the following parameters.
        rel_tolerance         --> we converge when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance
        abs_tolerance         --> we converge when sqrt(g,g) <= abs_tolerance
        max_iter              --> maximum number of iterations
        inner_rel_tolerance   --> relative tolerance used for the solution of the
                                  forward, adjoint, and incremental (fwd,adj) problems
        c_armijo              --> Armijo constant for sufficient reduction
        max_backtracking_iter --> Maximum number of backtracking iterations
        print_level           --> Print info on screen
        """
        self.model = model
        
        self.parameters = {}
        self.parameters["rel_tolerance"]         = 1e-6
        self.parameters["abs_tolerance"]         = 1e-12
        self.parameters["max_iter"]              = 20
        self.parameters["inner_rel_tolerance"]   = 1e-9
        self.parameters["c_armijo"]              = 1e-4
        self.parameters["max_backtracking_iter"] = 10
        self.parameters["print_level"]           = 0
        self.parameters["alpha"]                 = 1.
        
        self.it = 0
        self.converged = False
        self.ncalls = 0
        self.reason = 0
        self.final_grad_norm = 0
        
    def solve(self,a0):
        """
        Solve the constrained optimization problem with initial guess a0.
        Return the solution [u,a,p] 
        """
        rel_tol = self.parameters["rel_tolerance"]
        abs_tol = self.parameters["abs_tolerance"]
        max_iter = self.parameters["max_iter"]
        innerTol = self.parameters["inner_rel_tolerance"]
        c_armijo = self.parameters["c_armijo"]
        max_backtracking_iter = self.parameters["max_backtracking_iter"]
        alpha = self.parameters["alpha"]
        print_level = self.parameters["print_level"]
        
        [u,a,p] = self.model.generate_vector()
        self.model.solveFwd(u, [u, a0, p], innerTol)
        
        self.it = 0
        self.converged = False
        self.ncalls += 1
        
        ahat = self.model.generate_vector(PARAMETER)    
        mg = self.model.generate_vector(PARAMETER)
        
        cost_old, _, _ = self.model.cost([u,a0,p])
                
        while (self.it < max_iter) and (self.converged == False):
            self.model.solveAdj(p, [u,a0,p], innerTol)
            
            gradnorm = self.model.evalGradientParameter([u,a0,p], mg)
            
            if self.it == 0:
                gradnorm_ini = gradnorm
                tol = max(abs_tol, gradnorm_ini*rel_tol)
                
            # check if solution is reached
            if (gradnorm < tol) and (self.it > 0):
                self.converged = True
                self.reason = 1
                break
            
            self.it += 1
                        
            self.model.Rsolver().solve(ahat, -mg)
            
            alpha *= 2.
            descent = 0
            n_backtrack = 0
            
            mg_ahat = mg.inner(ahat)
                        
            while descent == 0 and n_backtrack < max_backtracking_iter:
                # update a and u
                a.zero()
                a.axpy(1., a0)
                a.axpy(alpha, ahat)
                self.model.solveFwd(u, [u, a, p], innerTol)
                
                cost_new, reg_new, misfit_new = self.model.cost([u,a,p])
                                
                # Check if armijo conditions are satisfied
                if (cost_new < cost_old + alpha * c_armijo * mg_ahat):
                    cost_old = cost_new
                    descent = 1
                    a0.zero()
                    a0.axpy(1., a)
                else:
                    n_backtrack += 1
                    alpha *= 0.5
                            
            if(print_level >= 0) and (self.it == 1):
                print "\n{0:3} {1:15} {2:15} {3:15} {4:15} {5:15}".format(
                      "It", "cost", "misfit", "reg", "||g||L2", "alpha")
                
            if print_level >= 0:
                print "{0:3d} {1:15e} {2:15e} {3:15e} {4:15e} {5:15e}".format(
                        self.it, cost_new, misfit_new, reg_new, gradnorm, alpha)
                
            if n_backtrack == max_backtracking_iter:
                self.converged = False
                self.reason = 2
                break
                                        
        self.final_grad_norm = gradnorm
        self.final_cost      = cost_new
        return [u,a0,p]