'''
Created on Mar 2, 2016

@author: uvilla
'''

import dolfin as dl
import numpy as np
from hippylib import *

from nse_non_linear_problem import NonlinearProblem
from nse_NewtonBacktrack import NewtonBacktrack
from nse_residual_norms import ResNormAinv

class RANS_AlgModel_Problem(PDEProblem):
    """
    This class provides an implementation of the abstract hIPPYlib class PDEProblem
    specific for the RANS equations with algebraic closure model.
    
    It provides methods to solve the (nonlinear) forward problem, the (linear) adjoint problem,
    and the incremental forward and adjoint problems.
    
    This class can be used:
    - to solve the inverse problem (see hIPPYlib Model)
    - to perform forward propagation of uncertainty (see ReducedQOI)
    
                             
    In what follows *u* denotes the state, *a* denotes the uncertain parameter, *p* the adjoint variable.
    """
    def __init__(self, Vh, model, bcs, bcs0, extra_args, parameter_loc_in_extra_args, verbose = True):
        """
        Constructor:
        - Vh: a list of finite element space for the state, parameter and adjoint variables.
              Note that the space for the state and adjoint variables need to be the same.
        - model: a class that provides the weak form for the residual and Jacobian of the forward problem.
                 See RANS_alg_model.py for an example.
        - bcs: list of essential boundary conditions for the forward problem.
        - bcs0: list of (homogeneus) essential boundary conditions for the adjoint and incremental problems.
        - initial_guess: an initial guess vector to speed-up the computations
        """
        mpi_comm = Vh[STATE].mesh().mpi_comm()
        self.mpi_rank = dl.MPI.rank(mpi_comm)
        self.mpi_size = dl.MPI.size(mpi_comm)
        
        self.Vh = Vh
        self.model = model
        self.bcs = bcs
        self.bcs0 = bcs0
        
        self.verbose = verbose
                
        self.A  = []
        self.Asolver = dl.PETScLUSolver()
        self.C = []
        self.Wum = []
        self.Wmm = []
        self.Wuu = []
        
        self.continuation_i = 0
        self.continuation_maxlen = 20
        self.continuation_states = []
        self.continuation_ms = []
        self.target_abs_tol = 1e-10
        
        self.fcp = dl.parameters["form_compiler"]
        self.fcp["quadrature_degree"] = 3
        #self.fcp["representation"] = "quadrature"
        #self.fcp["optimize"] = True
        #self.fcp["cpp_optimize"] = True
        #self.fcp["cpp_optimize_flags"] = "-O3"
        
        self.problem = NonlinearProblem(self.Vh[STATE], self.model, self.bcs, self.bcs0)
        self.problem.fcp = self.fcp
        weightsL2 = [dl.Constant(1.), dl.Constant(1.), dl.Constant(1.)]
        weightsH1 = [dl.Constant(1.), dl.Constant(1.), dl.Constant(1.)]
        self.W_fwd = self.problem.Norm(coeffsL2=weightsL2, coeffsH1=weightsH1)
        self.Wr_fwd = ResNormAinv(self.W_fwd)
        
        self.parameter_location = parameter_loc_in_extra_args 
        self.extra_args = extra_args
        
                        
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
        
    def initial_solve(self, state, x, tol):
        self.extra_args[self.parameter_location] = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        self.extra_args[-1] = False
        solver = NewtonBacktrack(self.problem, Wr=self.Wr_fwd, extra_args=tuple(self.extra_args) )
        if self.mpi_rank == 0:
            solver.parameters["print_level"] = 1
        else:
            solver.parameters["print_level"] = -1
        solver.parameters["max_iter"] = 100
        solver.parameters["rel_tolerance"] = 1e-6
        
        solver.solve(state)
        solver.parameters["abs_tolerance"] = solver.initial_norm * tol
        
        self.extra_args[-1] = True
        solver.extra_args=tuple(self.extra_args)
        solver.solve(state)
    
        assert solver.converged
        self.target_abs_tol = solver.final_norm
        assert len(self.continuation_ms) == 0
        self.continuation_states.append( state.copy() )
        self.continuation_ms.append( x[PARAMETER].copy() )
        
        self.extra_args[self.parameter_location] = None
                    
    def solveFwd(self, state, x, tol):
        """ Solve the possibly nonlinear Fwd Problem:
        Given a, find u such that
        \delta_p F(u,a,p;\hat_p) = 0 \for all \hat_p"""
           
        dm_norms = [ (x[PARAMETER] - mi).norm("l2") for mi in self.continuation_ms]
        
        min_dm_norms, idx = min( (dm_norms[ii], ii) for ii in range(len(self.continuation_ms)))
        
        m_start = self.continuation_ms[idx].copy()
        state.zero()
        state.axpy(1., self.continuation_states[idx])
        
        while (x[PARAMETER]-m_start).norm("l2") > dl.DOLFIN_EPS:
            self._fwd_cont(state, m_start, x[PARAMETER])
        
        if len(self.continuation_ms) < self.continuation_maxlen:
            self.continuation_states.append(state.copy())
            self.continuation_ms.append(x[PARAMETER].copy())
        else:
            index = (self.continuation_i % (self.continuation_maxlen-1)) +1
            self.continuation_states[index] = state.copy()
            self.continuation_ms[index] = x[PARAMETER].copy()
            self.continuation_i = index 
            
        
    def _fwd_cont(self, state, m, m_target):
        assert self.extra_args[-1] == True
        dm = m_target - m
        d_state, dd_state = self.forward_sensitivities([state, m], dm)
        m_new = self.generate_parameter()
        state_new = self.generate_state()
        alpha = 1.
        use_quad_approx = False
        while alpha > 1e-3:
            m_new.zero()
            m_new.axpy(1.-alpha, m)
            m_new.axpy(alpha, m_target)
            state_new.zero()
            state_new.axpy(1., state)
            state_new.axpy(alpha, d_state)
            m_new_fun = vector2Function(m_new, self.Vh[PARAMETER])
            self.extra_args[self.parameter_location] = m_new_fun
            r_lin = self.problem.residual(state_new, extra_args=tuple(self.extra_args))
            norm_r_lin = self.Wr_fwd.norm(r_lin)
            state_new.axpy(.5*alpha*alpha, dd_state)
            r_quad = self.problem.residual(state_new, extra_args=tuple(self.extra_args))
            norm_r_quad = self.Wr_fwd.norm(r_quad)
            
            if norm_r_quad < norm_r_lin:
                use_quad_approx = True
                break
            elif norm_r_lin < self.target_abs_tol:
                break
            else:
                alpha *= .5
                
        solver = NewtonBacktrack(self.problem, Wr=self.Wr_fwd, extra_args=tuple(self.extra_args))
        solver.parameters["print_level"] = -1
        solver.parameters["max_iter"] = 100
        solver.parameters["abs_tolerance"] = self.target_abs_tol
        
        state.axpy(alpha, d_state)
        if use_quad_approx:
            state.axpy(.5*alpha*alpha, dd_state)
        solver.solve(state)
        
        if not solver.converged:
            print "Not converged in {0} iterations. Final norm is {1}.".format(solver.final_it,solver.final_norm)
            
        m.zero()
        m.axpy(1., m_new)
        
        self.extra_args[self.parameter_location] = None
        
    def getJsolver(self,sm):
        assert self.extra_args[-1] == True
                
        [state_fun, m_fun] = [vector2Function(sm[ii], self.Vh[ii]) for ii in range(2)]        
        test = dl.TestFunction(self.Vh[STATE])
        trial = dl.TrialFunction(self.Vh[STATE])
        
        self.extra_args[self.parameter_location] = m_fun
        
        res_form = self.model.residual(state_fun, test, *self.extra_args)
        
        J_form = dl.derivative(res_form, state_fun, trial)
        b1_form = dl.inner(state_fun, test)*dl.dx
        
        J, dummy = dl.assemble_system(J_form, b1_form, bcs = self.bcs0, form_compiler_parameters = self.fcp)
        Jsolver = dl.PETScLUSolver()
        Jsolver.set_operator(J)
        
        self.extra_args[self.parameter_location] = None
        return Jsolver

            
    def forward_sensitivities(self, sm, dm, Jsolver = None):
        
        assert self.extra_args[-1] == True
        
        d_state = self.generate_state()
        dd_state = self.generate_state()
        
        [state_fun, m_fun] = [vector2Function(sm[ii], self.Vh[ii]) for ii in range(2)]
        dm_fun = vector2Function(dm, self.Vh[PARAMETER])
        
        test = dl.TestFunction(self.Vh[STATE])
        trial = dl.TrialFunction(self.Vh[STATE])
        
        self.extra_args[self.parameter_location] = m_fun
        
        res_form = self.model.residual(state_fun, test, *self.extra_args)
        b1_form = dl.derivative(res_form, m_fun, dm_fun)
        
        if Jsolver is None:
            J_form = dl.derivative(res_form, state_fun, trial)
            J, b1 = dl.assemble_system(J_form, b1_form, bcs = self.bcs0, form_compiler_parameters = self.fcp)
            Jsolver = dl.PETScLUSolver()
            Jsolver.set_operator(J)
        else:
            b1 = dl.assemble(b1_form, form_compiler_parameters=self.fcp)
            [bc.apply(b1) for bc in self.bcs0]
        
        Jsolver.solve(d_state, -b1)
        
        d_state_fun = vector2Function(d_state, self.Vh[STATE])
        
        b2_form = dl.derivative( dl.derivative(res_form, state_fun, d_state_fun), state_fun, d_state_fun ) \
                 + dl.Constant(2.)*dl.derivative( dl.derivative(res_form, state_fun, d_state_fun), m_fun, dm_fun ) \
                 + dl.derivative( dl.derivative(res_form, m_fun, dm_fun), m_fun, dm_fun )
                 
        b2 = dl.assemble(b2_form, form_compiler_parameters=self.fcp)
        [bc.apply(b2) for bc in self.bcs0]
        
        Jsolver.solve(dd_state, -b2)
        
        self.extra_args[self.parameter_location] = None
        
        return d_state, dd_state        
        
            
        
        
    def solveAdj(self, adj, x, adj_rhs, tol):
        """ Solve the linear Adj Problem: 
            Given a, u; find p such that
            \delta_u F(u,a,p;\hat_u) = 0 \for all \hat_u
        """
        assert self.extra_args[-1] == True
        
        problem = NonlinearProblem(self.Vh[STATE], self.model, self.bcs, self.bcs0)
        problem.fcp = self.fcp
        self.extra_args[self.parameter_location] = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        A = problem.Jacobian(x[STATE], extra_args=tuple(self.extra_args))
        problem.applyBC0(adj_rhs)
        solver = dl.PETScLUSolver()
        solver.set_operator(A)
        solver.solve_transpose(x[ADJOINT], adj_rhs)
        
        self.extra_args[self.parameter_location] = None
        
     
    def eval_da(self, x, out):
        assert self.extra_args[-1] == True
        """Given u,a,p; eval \delta_a F(u,a,p; \hat_a) \for all \hat_a """
        state_fun = vector2Function(x[STATE], self.Vh[STATE])
        param_fun = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        adj_fun   = vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        
        self.extra_args[self.parameter_location] = param_fun
        func = self.model.residual(state_fun, adj_fun, *self.extra_args)
        
        out.zero()
        dl.assemble(dl.derivative(func, param_fun), tensor=out, form_compiler_parameters = self.fcp)
        self.extra_args[self.parameter_location] = None
         
    def setLinearizationPoint(self,x):
        assert self.extra_args[-1] == True
        """ Set the values of the state and parameter
            for the incremental Fwd and Adj solvers """
        state_fun = vector2Function(x[STATE], self.Vh[STATE])
        param_fun = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        adjoint_fun = vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        
        x_fun = [state_fun,param_fun, adjoint_fun]
        x_test = [dl.TestFunction(self.Vh[STATE]),
                  dl.TestFunction(self.Vh[PARAMETER]),
                  dl.TestFunction(self.Vh[ADJOINT])]
        x_trial = [dl.TrialFunction(self.Vh[STATE]),
                  dl.TrialFunction(self.Vh[PARAMETER]),
                  dl.TrialFunction(self.Vh[ADJOINT])]
        
        self.extra_args[self.parameter_location] = param_fun
        f_form = self.model.residual(state_fun, adjoint_fun, *self.extra_args )
        
        g_form = [None,None,None]
        g_form[STATE] = dl.derivative(f_form, x_fun[STATE], x_test[STATE])
        g_form[PARAMETER] = dl.derivative(f_form, x_fun[PARAMETER], x_test[PARAMETER])
        g_form[ADJOINT] = dl.derivative(f_form, x_fun[ADJOINT], x_test[ADJOINT])

        
        Aform = dl.derivative(g_form[ADJOINT],state_fun, x_trial[STATE])
        self.A, dummy = dl.assemble_system(Aform, g_form[ADJOINT], self.bcs0, form_compiler_parameters = self.fcp)
        self.C = dl.assemble(dl.derivative(g_form[ADJOINT],param_fun, x_trial[PARAMETER]),form_compiler_parameters = self.fcp )
        [bc.zero(self.C) for bc in self.bcs0]
        self.Wum = dl.assemble(dl.derivative(g_form[STATE],param_fun, x_trial[PARAMETER]), form_compiler_parameters = self.fcp)
        [bc.zero(self.Wum) for bc in self.bcs0]
        
        Wuuform = dl.derivative(g_form[STATE],state_fun, x_trial[STATE])
        self.Wuu, dummy = dl.assemble_system(Wuuform, g_form[STATE], self.bcs0, form_compiler_parameters = self.fcp)
        [bc.zero(self.Wuu) for bc in self.bcs0]

        self.Wmm = dl.assemble(dl.derivative(g_form[PARAMETER],param_fun, x_trial[PARAMETER]), form_compiler_parameters = self.fcp)
        
        self.Asolver.set_operator(self.A)
        self.extra_args[self.parameter_location] = None
        
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
        KKT[PARAMETER, PARAMETER] = self.Wmm
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
        
