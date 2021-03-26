__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-08-30"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"

"""
Base class for all Turbulence model solvers.
"""
from cbc.pdesys.PDESystem import *

solver_parameters = copy.deepcopy(default_solver_parameters)
solver_parameters = recursive_update(default_solver_parameters, {
    'apply': defaultdict(lambda: 'use_formula'),
    'familyname': 'Turbulence model'
})

class TurbSolver(PDESystem):
    
    def __init__(self, system_composition, problem, parameters):
        PDESystem.__init__(self, system_composition, problem, parameters)

    def setup(self):
        PDESystem.setup(self)        
        self.nu = self.problem.NS_solver.nuM
        # Get the boundaries from NS_problem.
        self.u_ = self.problem.NS_solver.u_
        self.problem.initialize(self)
        self.boundaries = self.problem.boundaries
        self.model_parameters()
        self.bc = self.create_BCs(self.boundaries)      
        self.define()
        
    def define(self):
        self.problem.NS_solver.nu = self.problem.NS_solver.nuM + self.nut_
        self.problem.NS_solver.nut_ = self.nut_
        self.problem.NS_solver.define()
            
    def model_parameters(self):
        pass
