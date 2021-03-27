'''
Created on Mar 2, 2016

@author: uvilla

Modified on July 9, 2016 by slan
Modified on August 25, 2016 by slan
Modified on Feburary 8, 2017 by slan
'''

import dolfin as dl
import numpy as np
from hippylib import *

from nse_non_linear_problem import NonlinearProblem
from nse_NewtonBacktrack import NewtonBacktrack
from nse_residual_norms import ResNormAinv

# add a counter wrapper to count certain function calls
from functools import wraps
def counter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.count+= 1
        return func(*args, **kwargs)
    wrapper.count= 0
    wrapper.__name__= func.__name__
    return wrapper

class RANSProblem(PDEProblem):
    """
    This class provides an implementation of the abstract hIPPYlib class PDEProblem
    specific for the RANS equations with k-e closure model.
    
    It provides methods to solve the (nonlinear) forward problem, the (linear) adjoint problem,
    and the incremental forward and adjoint problems.
    
    This class can be used:
    - to solve the inverse problem (see hIPPYlib Model)
    - to perform forward propagation of uncertainty (see ReducedQOI)
    
    To solve the forward problem, two different continuation algorithms are provided:
    - TransientContinuation: it solves a time transient simulation that asymptotically approaches the steady
                             state solution.
                             This is the solver of choice when computing the solution from scratch.
    - ParameterContinuation: given a solution pair (state, parameter) and a new point in parameter space, computes
                             the new state using a Predictor-Corrector continuation algorithms.
                             This solver is particularly efficient when the difference between the old and new parameter is small.
                             
    In what follows *u* denotes the state, *a* denotes the uncertain parameter, *p* the adjoint variable.
    """
    def __init__(self, Vh, model, bcs, bcs0, initial_guess, verbose = True):
        """
        Constructor:
        - Vh: a list of finite element space for the state, parameter and adjoint variables.
              Note that the space for the state and adjoint variables need to be the same.
        - model: a class that provides the weak form for the residual and Jacobian of the forward problem.
                 See RANSModel_inadeguate_Cmu for an example.
        - bcs: list of essential boundary conditions for the forward problem.
        - bcs0: list of (homogeneus) essential boundary conditions for the adjoint and incremental problems.
        - initial_guess: an initial guess vector for the TransientContinuation solver
        """
        self.Vh = Vh
        self.model = model
        self.bcs = bcs
        self.bcs0 = bcs0
        
        self.initial_guess = initial_guess.copy()

        self.verbose = verbose
        
        self.x_cont = None
        self.x_latest = None
        self.final_norm_cont = 1e-10
        
        self.A  = []
        self.Asolver = dl.PETScLUSolver()
        self.C = []
        self.Wum = []
        self.Wmm = []
        self.Wuu = []
        
        self.fcp = {}
        self.fcp["quadrature_degree"] = 3
        #self.fcp["representation"] = "quadrature"
        #self.fcp["optimize"] = True
        #self.fcp["cpp_optimize"] = True
        #self.fcp["cpp_optimize_flags"] = "-O3"
    
    def setTransientContinuationSolver(self):
        """
        Set the class to use the Transient Continuation Solver.
        """
        self.x_cont = None
          
    def setParameterContinuationSolver(self,x_cont, final_norm):
        """
        Set the class to use the Parameter Continuation Solver.
        - x_cont = [state, parameter] represent a particular solution pair.
        - final_norm is the desidered absolute residual norm for the continuation algorithm.
        """
        self.x_cont = [x_cont[STATE].copy(), x_cont[PARAMETER].copy() ]
        self.final_norm_cont = final_norm
                
    def generate_state(self):
        """ return a vector in the shape of the state """
        return dl.Function(self.Vh[STATE]).vector()
    
    def generate_parameter(self):
        """ return a vector in the shape of the parameter """
        return dl.Function(self.Vh[PARAMETER]).vector()
    
    def init_parameter(self, m):
        """ initialize the parameter """
        dummy = self.generate_parameter()
        m.init( dummy.mpi_comm(), dummy.local_range() )
            
    @counter
    def solveFwd(self, state, x, tol):
        """ Solve the possibly nonlinear Fwd Problem:
        Given a, find u such that
        \delta_p F(u,a,p;\hat_p) = 0 \for all \hat_p"""
        
        if self.x_cont is None:
            self._solveFwdTransientCont(state, x, tol)
        else:
            self._solveFwdParamCont(state, x, tol)
            
        self.x_latest = [state.copy(), x[PARAMETER].copy()]
            
    def _solveFwdParamCont(self, state, x,tol):
        """
        The predictor-corrector parameter continuation solver.
        """
        
        m_hat1 = x[PARAMETER] - self.x_cont[PARAMETER]
        m_hat1_norm = m_hat1.norm("l2")
        
        if self.x_latest is not None:
            m_hat2 = x[PARAMETER] - self.x_latest[PARAMETER]
            m_hat2_norm = m_hat2.norm("l2")
        else:
            m_hat2_norm = 1. + m_hat1_norm
        
        if m_hat1_norm < m_hat2_norm:
            x_cont = self.x_cont
        else:
            x_cont = self.x_latest
        
        state.zero()
        state.axpy(1., x_cont[STATE])
        
        m_hat = x[PARAMETER] - x_cont[PARAMETER]
        m = self.generate_parameter()
        new_state = self.generate_state()
        new_state.axpy(1., x_cont[STATE])
        dstate = self.generate_state()
        
        if m_hat.norm("l2") < 1e-12:
            return
        
        problem = NonlinearProblem(self.Vh[STATE], self.model, self.bcs, self.bcs0)
        problem.fcp = self.fcp
        W = problem.Norm(coeffsL2=[dl.Constant(1e-2) for i in range(4)], coeffsH1=[dl.Constant(1.) for i in range(4)])
        Wr = ResNormAinv(W)
        
        alpha = .0
        dalpha = 0.01
        
        total_iter = 0
        total_reject = 0
        
        while 1.:
            # Tangent Step
            self._solveTangentProblem(dstate, m_hat, [state, m])
            m.zero()
            m.axpy(1., x_cont[PARAMETER])
            if alpha + dalpha < 1.:
                m.axpy(alpha+dalpha, m_hat)
                new_state.axpy(dalpha, dstate)
            else:
                m.axpy(1., m_hat)
                new_state.axpy(alpha - 1., dstate)
            # Newton Correction
            extra_args = (dl.Constant(0.), vector2Function(m, self.Vh[PARAMETER]), None, None, True)
            solver = NewtonBacktrack(problem, Wr=Wr, extra_args=extra_args)
            solver.parameters["abs_tolerance"] = self.final_norm_cont
            solver.parameters["max_iter"] = 20
            solver.parameters["print_level"] = -1
            solver.solve(new_state)
            if self.verbose:
                print "{0:5e} {1:5e} {2:10e} {3:5d} {4}".format(alpha, dalpha, solver.final_norm, solver.final_it, solver.converged)
            total_iter += solver.final_it + 1
            if solver.converged == False and dalpha == 1e-4:
                self._solveFwdTransientCont(new_state,x, tol)
                state.zero()
                state.axpy(1., new_state)
                alpha += dalpha
            elif solver.converged:
                state.zero()
                state.axpy(1., new_state)
                alpha += dalpha
                if solver.final_it < 3:
                    dalpha *= 2.
                elif solver.final_it > 5:
                    dalpha *= .9
            else:
                total_reject += 1
                new_state.zero()
                new_state.axpy(1., state)
                dalpha *= .5               
                
            dalpha = max(dalpha, 1e-4)
                
            if alpha >= 1.:
                break
        
        if self.verbose:    
            print "Total iter: ", total_iter, "Total rejects: ", total_reject
    
    def _solveTangentProblem(self, dstate, m_hat, x):
        """
        The predictor step for the parameter continuation algorithm.
        """
        state_fun = vector2Function(x[STATE], self.Vh[STATE])
        statel_fun = vector2Function(x[STATE], self.Vh[STATE])
        param_fun = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        param_hat_fun = vector2Function(m_hat, self.Vh[PARAMETER])
        x_test   = [dl.TestFunction(self.Vh[STATE]), dl.TestFunction(self.Vh[PARAMETER])]
        x_trial  = [dl.TrialFunction(self.Vh[STATE]), dl.TrialFunction(self.Vh[PARAMETER])]
        
        res = self.model.residual(state_fun, x_test[STATE], dl.Constant(0.), param_fun, None, statel_fun, True )
        Aform = dl.derivative(res,state_fun, x_trial[STATE]) + dl.derivative(res,statel_fun, x_trial[STATE])
        b_form = dl.derivative(res,param_fun, param_hat_fun)
        
        A, b = dl.assemble_system(Aform, b_form, self.bcs0, form_compiler_parameters = self.fcp)
        solver = dl.PETScLUSolver()
        solver.set_operator(A)
        solver.solve(dstate, -b)

     
    def _solveFwdTransientCont(self,state,x, tol):   
        """
        The time transient continuation solver
        """
        
        problem = NonlinearProblem(self.Vh[STATE], self.model, self.bcs, self.bcs0)
        problem.fcp = self.fcp
        W = problem.Norm(coeffsL2=[dl.Constant(1e-2) for i in range(4)], coeffsH1=[dl.Constant(1.) for i in range(4)])
        Wr = ResNormAinv(W)
       
        state_0 = self.generate_state()
        state_0.axpy(1., x[STATE])

        state.zero()
        state.axpy(1., state_0)
 
        extra_args = (dl.Constant(0.),
                      vector2Function(x[PARAMETER], self.Vh[PARAMETER]),
                      None,
                      None,
                      True)
        norm_steady_state_r0 = Wr.norm( problem.residual(state, extra_args) )
        norm_steady_state_r = norm_steady_state_r0
        
        atol = 1e-10
        rtol = tol
        tol = max(atol, rtol*norm_steady_state_r0)
        if norm_steady_state_r < 5e-3:
            sigma = 0.0
        else:
            sigma = 1.
        t = 0.
        
        if self.verbose:
            print "{0:5s} {1:5s} {2:15s}".format("t", "sigma", "r")
            print "{0:5e} {1:5e} {2:15e}".format(t, sigma, norm_steady_state_r)
        
        it = 0.
        while norm_steady_state_r > tol:
            if sigma == 0:
                extra_args = (dl.Constant(sigma),
                              vector2Function(x[PARAMETER], self.Vh[PARAMETER]),
                              None,
                              None,
                              True
                              )
            else:
                extra_args = (dl.Constant(sigma),
                              vector2Function(x[PARAMETER], self.Vh[PARAMETER]),
                              vector2Function(state_0, self.Vh[STATE]),
                              vector2Function(state_0, self.Vh[STATE]),
                              False
                              )
            solver = NewtonBacktrack(problem, Wr=Wr, extra_args=extra_args)
            if sigma > 0:
                solver.parameters["print_level"] = -1
                solver.parameters["max_iter"] = 20
            else:
                solver.parameters["print_level"] = 1
                solver.parameters["max_iter"] = 100
                
            if not self.verbose:
                solver.parameters["print_level"] = -1
                
            solver.solve(state)
            state_0.zero()
            state_0.axpy(1., state)
            
            extra_args = (dl.Constant(0.),
                      vector2Function(x[PARAMETER], self.Vh[PARAMETER]),
                      None,
                      None,
                      True)
            norm_steady_state_r = Wr.norm( problem.residual(state, extra_args) )
            it = it + 1
            if sigma == 0:
                t = np.Inf
            else:
                t = t + 1./sigma
            if self.verbose:
                print "{0:5e} {1:5e} {2:15e} {3:3} {4}".format(t, sigma, norm_steady_state_r,
                                                               solver.final_it, solver.converged)
                
            if sigma == 0.:
                break
            if solver.converged:
                sigma = 0.87*sigma
                
            if sigma < 1e-6 or norm_steady_state_r < 1e-4:
                sigma = 0.
            #raise RuntimeError if the solver blows up
            if norm_steady_state_r > 1e1 and solver.final_it == 0 and not solver.converged:
                raise RuntimeError('The forward solver blows up!')
                
        if norm_steady_state_r > tol and self.verbose:
            print "Forward solver failed. Final norm is", norm_steady_state_r, "tolerance was", tol

        
        
    @counter
    def solveAdj(self, adj, x, adj_rhs, tol):
        """ Solve the linear Adj Problem: 
            Given a, u; find p such that
            \delta_u F(u,a,p;\hat_u) = 0 \for all \hat_u
        """
        problem = NonlinearProblem(self.Vh[STATE], self.model, self.bcs, self.bcs0)
        problem.fcp = self.fcp
        extra_args = (dl.Constant(0.),
                      vector2Function(x[PARAMETER], self.Vh[PARAMETER]),
                      None, None, True )
        A = problem.Jacobian(x[STATE], extra_args)
        problem.applyBC0(adj_rhs)
        solver = dl.PETScLUSolver()
        solver.set_operator(A)
        solver.solve_transpose(x[ADJOINT], adj_rhs)
        
     
    def eval_da(self, x, out):
        """Given u,a,p; eval \delta_a F(u,a,p; \hat_a) \for all \hat_a """
        state_fun = vector2Function(x[STATE], self.Vh[STATE])
        param_fun = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        adj_fun   = vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        
        func = self.model.residual(state_fun, adj_fun,
                      dl.Constant(0.),
                      param_fun, None, None, True )
        
        out.zero()
        dl.assemble(dl.derivative(func, param_fun), tensor=out, form_compiler_parameters = self.fcp)
         
    def setLinearizationPoint(self,x):
        """ Set the values of the state and parameter
            for the incremental Fwd and Adj solvers """
        state_fun = vector2Function(x[STATE], self.Vh[STATE])
        statel_fun = vector2Function(x[STATE], self.Vh[STATE])
        param_fun = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        adjoint_fun = vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        
        x_fun = [state_fun,param_fun, adjoint_fun]
        x_test = [dl.TestFunction(self.Vh[STATE]),
                  dl.TestFunction(self.Vh[PARAMETER]),
                  dl.TestFunction(self.Vh[ADJOINT])]
        x_trial = [dl.TrialFunction(self.Vh[STATE]),
                  dl.TrialFunction(self.Vh[PARAMETER]),
                  dl.TrialFunction(self.Vh[ADJOINT])]
        
        f_form = self.model.residual(state_fun, adjoint_fun,
                      dl.Constant(0.), param_fun, None, statel_fun, True )
        
        g_form = [None,None,None]
        g_form[STATE] = dl.derivative(f_form, x_fun[STATE], x_test[STATE]) + dl.derivative(f_form, statel_fun, x_test[STATE])
        g_form[PARAMETER] = dl.derivative(f_form, x_fun[PARAMETER], x_test[PARAMETER])
        g_form[ADJOINT] = dl.derivative(f_form, x_fun[ADJOINT], x_test[ADJOINT])

        
        Aform = dl.derivative(g_form[ADJOINT],state_fun, x_trial[STATE]) + dl.derivative(g_form[ADJOINT],statel_fun, x_trial[STATE]) 
        self.A, dummy = dl.assemble_system(Aform, g_form[ADJOINT], self.bcs0, form_compiler_parameters = self.fcp)
        self.C = dl.assemble(dl.derivative(g_form[ADJOINT],param_fun, x_trial[PARAMETER]),form_compiler_parameters = self.fcp )
        [bc.zero(self.C) for bc in self.bcs0]
        self.Wum = dl.assemble(dl.derivative(g_form[STATE],param_fun, x_trial[PARAMETER]), form_compiler_parameters = self.fcp)
        [bc.zero(self.Wum) for bc in self.bcs0]
        
        Wuuform = dl.derivative(g_form[STATE],state_fun, x_trial[STATE]) + dl.derivative(g_form[STATE],statel_fun, x_trial[STATE])
        self.Wuu, dummy = dl.assemble_system(Wuuform, g_form[STATE], self.bcs0, form_compiler_parameters = self.fcp)
        [bc.zero(self.Wuu) for bc in self.bcs0]

        self.Waa = dl.assemble(dl.derivative(g_form[PARAMETER],param_fun, x_trial[PARAMETER]), form_compiler_parameters = self.fcp)
        
        self.Asolver.set_operator(self.A)
        
    @counter
    def solveIncremental(self, out, rhs, is_adj, mytol):
        """ If is_adj = False:
            Solve the forward incremental system:
            Given u, a, find \tilde_u s.t.:
            \delta_{pu} F(u,a,p; \hat_p, \tilde_u) = rhs for all \hat_p.
            
            If is_adj = True:
            Solve the adj incremental system:
            Given u, a, find \tilde_p s.t.:
            \delta_{up} F(u,a,p; \hat_u, \tilde_p) = rhs for all \delta_u.
        """
        if is_adj:
            self.Asolver.solve_transpose(out, rhs)
        else:
            self.Asolver.solve(out, rhs)
            
    
    def apply_ij(self,i,j, dir, out):   
        """
            Given u, a, p; compute 
            \delta_{ij} F(u,a,p; \hat_i, \tilde_j) in the direction \tilde_j = dir for all \hat_i
        """
        KKT = {}
        KKT[STATE,STATE] = self.Wuu
        KKT[STATE, PARAMETER] = self.Wum
        KKT[PARAMETER, PARAMETER] = self.Waa
        KKT[ADJOINT, STATE] = self.A
        KKT[ADJOINT, PARAMETER] = self.C
        
        if i == STATE and j == PARAMETER:
            KKT[STATE,PARAMETER].mult(dir,out)
        elif j == STATE and i == PARAMETER:
            KKT[STATE,PARAMETER].transpmult(dir,out)
        elif i >= j:
            KKT[i,j].mult(dir, out)
        else:
            KKT[j,i].transpmult(dir, out)
        
