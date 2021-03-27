__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2011-08-11"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU GPL version 3 or any later version"

"""
Newton solver for the stabilized Eikonal equation.
"""
from cbc.pdesys.PDESystem import *

solver_parameters = copy.deepcopy(default_solver_parameters)
solver_parameters = recursive_update(solver_parameters, {
    'eps' : Constant(0.02),
    'iteration_type': 'Newton',
    'time_integration': 'Steady',
    'max_iter': 100
})

class Eikonal(PDESystem):
    def __init__(self, mesh, boundaries, parameters=solver_parameters):
        PDESystem.__init__(self, [['y']], mesh, parameters)
        self.f = f = Constant(1.0)
        self.bc['y'] = self.create_BCs(boundaries, self.V['y'])
        
        # Initialize by solving similar elliptic problem
        u, v = self.qt['y'], self.vt['y']
        F1 = inner(grad(u), grad(v))*dx - f*v*dx
        a1, L1 = lhs(F1), rhs(F1)
        A1, b1 = assemble_system(a1, L1)
        for bc in self.bc['y']: bc.apply(A1, b1)
        solve(A1, self.y_.vector(), b1)
        #
        self.solve()

    def define(self):
        self.eps = self.prm['eps']
        self.pdesubsystems['y'] = eval('Eikonal_' + str(self.prm['pdesubsystem']['y'])) \
                                      (vars(self), ['y'], bcs=self.bc['y'])
                
    def create_BCs(self, bcs, V):
        """Create boundary conditions for Eikonal's equation based on 
        boundaries in list bcs. Assigns homogeneous Dirichlet boundary 
        conditions on walls. """
        bcu = []
        for bc in bcs:
            if bc.type() in ('Wall'):
                add_BC(bcu, V, bc, Constant(0.))
        return bcu
        
class Eikonal_1(PDESubSystem):

    def form(self, y, y_, v_y, f, eps, **kwargs):
        return sqrt(inner(grad(y_), grad(y_)))*v_y*dx  -  f*v_y*dx + \
               eps*inner(grad(y_), grad(v_y))*dx

    def update(self):
        bound(self.x)
