__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-10-30"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"

from NSProblem import *
from numpy import arctan, array
import pylab as pp
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.special import erf

problem_parameters['pressure_bc'] = True
problem_parameters['Nx'] = 60
problem_parameters['Ny'] = 60
                
class U(Expression):
    """ Velocity """
    
    def __init__(self, **kwargs):
        self.dy = kwargs['dy']
        
    def eval(self, values, x):
        dy0 = self.dy(x[0])
        values[0] = -(x[1] - dy0)*(x[1] + dy0)
        values[1] = 0.
        if abs(x[1] + dy0) < 2.*DOLFIN_EPS or abs(x[1] - dy0) < 2.*DOLFIN_EPS:
            values[0] = 0.
            
    def value_shape(self):
        return (2,)

class U0(Expression):
    """ Velocity """
    
    def __init__(self, **kwargs):
        self.dy = kwargs['dy']
        
    def eval(self, values, x):
        dy0 = self.dy(x[0])
        values[0] = -(x[1] - dy0)*(x[1] + dy0)
        if abs(x[1] + dy0) < 2.*DOLFIN_EPS or abs(x[1] - dy0) < 2.*DOLFIN_EPS:
            values[0] = 0.
            
class UP(Expression):
    """Velocity and pressure."""    
    def __init__(self, **kwargs):
        self.dy = kwargs['dy']
        
    def eval(self, values, x):
        dy0 = self.dy(x[0])
        values[0] = -(x[1] - dy0)*(x[1] + dy0)
        values[1] = 0.
        values[2] = 0.
        if abs(x[1] + dy0) < 2*DOLFIN_EPS or abs(x[1] - dy0) < 2.*DOLFIN_EPS:
            values[0] = 0.
            
    def value_shape(self):
        return (3,)

class diffusor(NSProblem):
    """Expanding diffusor problem."""    
    def __init__(self, parameters):
        NSProblem.__init__(self, parameters=parameters)

        self.prm['viscosity'] = 1./self.prm['Re']

        # Use Expressions for 'complicated' profiles
        self.inlet_velocity = {'u' : U (dy=self.dy),
                               'u0': U0(dy=self.dy),
                               'u1': "0.0",
                               'p' : "0.0",
                               'up': UP(dy=self.dy)}
        
        # Otherwise just initialize to zero
        self.zero_velocity  = Initdict(u  = ("0.0", "0.0"),
                                       p  = "0.0")

        # Generate mesh and mark boundaries
        self.mesh, self.boundaries = self.create_mesh_and_boundaries()

        self.prm['dt'] = self.timestep()
        
        # Choose dictionary for initializing
        transient = self.prm['time_integration']=='Transient'
        self.q0 = self.zero_velocity if transient else self.inlet_velocity

    def create_mesh_and_boundaries(self):
        self.L = 16.
        m = Rectangle(0., -1., self.L, 1., self.prm['Nx'], self.prm['Ny'])
        x = m.coordinates()
        # Create stretched mesh in y-direction
        x[:, 1] = arctan(1.*pi*(x[:, 1]))/arctan(1.*pi) 
        self.mf = FacetFunction("uint", m)     # Facets
        self.mf.set_all(0)
        # We mark the boundaries before modifying the mesh:
        walls  = FlowSubDomain(lambda x, on_boundary: (near(x[1], 1.) or near(x[1], -1.)
                                                      and on_boundary),
                              bc_type = 'Wall',
                              mf  = self.mf)
            
        if self.prm['pressure_bc']:
            inlet  = FlowSubDomain(lambda x, on_boundary: near(x[0], 0.) and on_boundary,
                                bc_type = 'ConstantPressure',
                                func = {'p': Constant(2./self.prm['Re']*self.L*0.3)},
                                mf = self.mf)
            
            outlet = FlowSubDomain(lambda x, on_boundary: (abs(x[0]-self.L) < 10.*DOLFIN_EPS 
                                                        and on_boundary),
                                bc_type = 'ConstantPressure',
                                func = {'p': Constant(0.0)},
                                mf  = self.mf)
        else:
            inlet  = FlowSubDomain(lambda x, on_boundary: near(x[0], 0.) and on_boundary,
                                bc_type = 'VelocityInlet',
                                func = self.inlet_velocity,
                                mf = self.mf)
            
            outlet = FlowSubDomain(lambda x, on_boundary: (abs(x[0]-self.L) < 10.*DOLFIN_EPS 
                                                        and on_boundary),
                                bc_type = 'Outlet',
                                func = {'p': Constant(0.0)},
                                mf  = self.mf)
        
        # Now that the boundaries are marked we can create the diffusor by 
        # expanding the mesh
        x[:,1] = x[:, 1]*self.dy(x[:, 0])        
        return m, [walls, inlet, outlet]
        
    def dy(self, x, N=6.):
        return (erf(x - self.L/2.) + N)/(N - 1.)
        
    def __info__(self):
        return 'Axial diffusior'

if __name__ == '__main__':
    import cbc.cfd.icns as icns
    from cbc.cfd.icns import solver_parameters
    from time import time
    set_log_active(True)
    problem_parameters['time_integration']='Steady'
    problem_parameters['T'] = 1.
    problem_parameters['max_iter'] = 10
    problem_parameters['plot_velocity'] = False
    problem_parameters['pressure_bc'] = False
    solver_parameters = recursive_update(solver_parameters, 
    dict(degree=dict(u=1),
         pdesubsystem=dict(u=1, p=1, velocity_update=1, up=1), 
         linear_solver=dict(u='lu', p='lu', velocity_update='lu'), 
         precond=dict(u='jacobi', p='amg', velocity_update='jacobi'),
         iteration_type='Picard')
         )
    
    NS_problem = diffusor(problem_parameters)
    #NS_solver = icns.NSFullySegregated(NS_problem, solver_parameters)
    #NS_solver = icns.NSSegregated(NS_problem, solver_parameters)
    NS_solver = icns.NSCoupled(NS_problem, solver_parameters)
    #NS_solver.pdesubsystems['u1'].prm['monitor_convergence'] = True
    #NS_solver.pdesubsystems['u2'].prm['monitor_convergence'] = True
    #NS_solver.pdesubsystems['p'].prm['monitor_convergence'] = True
    #NS_solver.pdesubsystems['u1_update'].prm['monitor_convergence'] = True
    #NS_solver.pdesubsystems['u2_update'].prm['monitor_convergence'] = True
    t0 = time()
    NS_problem.solve()
    print 'time = ', time()-t0
    print summary()
    plot(NS_solver.u_)
    interactive()
    