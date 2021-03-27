"""This demo illustrates how to use DOLFIN and CBC.PDESys
to solve the Cahn-Hilliard equation, which is a time-dependent 
nonlinear PDE """

__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2011-06-30"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU GPL version 3 or any later version"

import random
from cbc.pdesys import *
from dolfin import parameters as dolfin_parameters
dolfin_parameters["form_compiler"]["optimize"]     = True
dolfin_parameters["form_compiler"]["cpp_optimize"] = True
dolfin_parameters["form_compiler"]["representation"] = "quadrature"

# Class representing the intial conditions
class InitialConditions(Expression):
    def __init__(self):
        random.seed(2 + MPI.process_number())
    def eval(self, values, x):
        values[0] = 0.63 + 0.02*(0.5 - random.random())
        values[1] = 0.0
    def value_shape(self):
        return (2,)

class CH_Problem(Problem):
    
    def __init__(self, parameters):
        Problem.__init__(self, parameters=parameters)
        self.mesh = UnitSquare(self.prm['N'], self.prm['N'])
        
    def update(self):
        plot(self.pdesystems['default'].c_, rescale=True)

class CH_Solver(PDESystem):
    
    def __init__(self, problem, parameters):
        PDESystem.__init__(self, [['c', 'mu']], problem, parameters=parameters)
        self.theta = Constant(problem.prm['theta'], cell=self.V['cmu'].cell())
        self.lmbda = Constant(problem.prm['lmbda'])
        
        # Compute the chemical potential df/dc
        c = variable(self.c)
        f = 100*c**2*(1-c)**2
        self.dfdc = diff(f, c)
        #self.dfdc = 200.*(c*(1.-c)**2 - c**2*(1.-c)) # Analytical form
        
        # Set up variational problem
        self.define()
        
    def define(self):
        self.pdesubsystems['cmu'] = eval('CH_' + 
               str(self.prm['pdesubsystem']['cmu']))(vars(self), ['c', 'mu'])
                
class CH_1(PDESubSystem):
    """Variational form of the Cahn Hilliard problem."""
    def form(self, c, v_c, c_1, mu, v_mu, mu_1, theta, dfdc, lmbda, dt, **kwargs):
        mu_mid = (1. - theta)*mu + theta*mu_1
        return inner(c - c_1, v_c)*dx + dt*inner(grad(mu_mid), grad(v_c))*dx \
          + inner(mu - dfdc, v_mu)*dx - lmbda*dot(grad(c), grad(v_mu))*dx

if __name__ == '__main__':
    # Update the default solver and problem parameters
    problem_parameters.update({
        'dt': 5.0e-6,                      # Timestep
        'lmbda': 1.0e-2,                   # Model parameter
        'theta': 0.5,                      # Model parameter
        'time_integration': 'Transient',   # Transient problem
        'T': 4.0e-4,                       # End time
        'N': 96                            # Mesh size
    })
    solver_parameters['iteration_type'] = 'Newton'
    problem = CH_Problem(problem_parameters)
    solver  = CH_Solver(problem, solver_parameters)
    problem.q0 = {'cmu': InitialConditions()}
    problem.initialize(solver)
    problem.solve(max_iter=6)
