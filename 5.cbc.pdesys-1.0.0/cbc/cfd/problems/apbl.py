__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-10-30"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"

from NSProblem import *
from numpy import arctan, array, loadtxt, zeros
from scipy.interpolate import InterpolatedUnivariateSpline as ius

problem_parameters['pressure_bc'] = True

class U0(Expression):
    """Velocity."""    
    def __init__(self, bump, **kwargs):
        self.bump = bump
        
    def eval(self, values, x):
        y0 = self.bump(x[0])
        values[0] = (x[1] - y0)*(2. - x[1])
        values[1] = 0.
        if abs(x[1] - y0) < 2*DOLFIN_EPS or abs(x[1] - 2.) < 2.*DOLFIN_EPS:
            values[0] = 0.
            
    def value_shape(self):
        return (2,)

class UP0(Expression):
    """Velocity and pressure."""    
    def __init__(self, bump, **kwargs):
        self.bump = bump
        
    def eval(self, values, x):
        y0 = self.bump(x[0])
        values[0] = (x[1] - y0)*(2. - x[1])
        values[1] = 0.
        values[2] = 0.
        if abs(x[1] - y0) < 2*DOLFIN_EPS or abs(x[1] - 2.) < 2.*DOLFIN_EPS:
            values[0] = 0.

    def value_shape(self):
        return (3,)

class apbl(NSProblem):
    """Set up laminar channel flow test case with a bump on the lower wall. 
    The channel spans  0 <= x <= 30 and 0 <= y <= 2         
    """    
    def __init__(self, parameters):
        NSProblem.__init__(self, parameters=parameters)        
        self.prm['viscosity'] = 1./self.prm['Re']
        self.L = 30.
        f = open('../data/Re395/statistics_streamwise.txt')
        a = loadtxt(f, comments='%')
        sz = a.shape
        a2 = zeros((sz[0] + 2, 2))
        a2[1:-1, :] = a[:, :2]
        a2[1:-1, 0] += 10. # Bump starts at x = 10
        a2[-1, 0] = self.L
        self.bump = ius(a2[:, 0], a2[:, 1]) # Create bump-spline
        
        # Use Expressions for 'complicated' profiles
        self.inlet_velocity = {'u': U0(self.bump),
                               'p': ('0'),
                               'up': UP0(self.bump)}
                               
        # Otherwise just initialize to zero
        self.zero_velocity  = Initdict(u  = ("0.0", "0.0"),
                                       p  = "0.0")
                                       
        self.mesh, self.boundaries = self.create_mesh_and_boundaries()
        
        self.prm['dt'] = self.timestep()
        
        transient = self.prm['time_integration']=='Transient'
        self.q0 = self.zero_velocity if transient else self.inlet_velocity

    def create_mesh_and_boundaries(self):
        m = Rectangle(0., -1., self.L, 1., self.prm['Nx'], self.prm['Ny'])
        
        # Create stretched mesh in y-direction
        x = m.coordinates()
        x[:, 1] = arctan(pi*(x[:, 1]))/arctan(pi) + 1. 
        
        # We have to mark the boundaries before modifying the mesh:
        self.mf = FacetFunction("uint", m)
        self.mf.set_all(0)
        
        walls = FlowSubDomain(lambda x, on_bnd: ((abs(x[1]) < 5.*DOLFIN_EPS or
                                                  abs(x[1]-2.) < 5.*DOLFIN_EPS)
                                                  and on_bnd),
                              bc_type = 'Wall',
                              mf = self.mf)
                             
        if self.prm['pressure_bc']:
            inlet = FlowSubDomain(lambda x, on_bnd: (near(x[0], 0.) and on_bnd),
                                  bc_type = 'ConstantPressure',
                                  func = {'p': 2./self.prm['Re']*self.L*1.25},
                                  mf = self.mf)
                                
            # near(x[0], self.L) does not work (roundoff?)
            outlet = FlowSubDomain(lambda x, on_bnd: (abs(x[0]-self.L) < 10.*DOLFIN_EPS 
                                                      and on_bnd),
                                   bc_type = 'ConstantPressure',
                                   func = {'p': Constant(0.0)},
                                   mf = self.mf)
        else:
            inlet = FlowSubDomain(lambda x, on_bnd: (near(x[0], 0.) and on_bnd),
                                  bc_type = 'VelocityInlet',
                                  func = self.inlet_velocity,
                                  mf = self.mf)
                                
            # near(x[0], self.L) does not work (roundoff?)
            outlet = FlowSubDomain(lambda x, on_bnd: (abs(x[0]-self.L) < 10.*DOLFIN_EPS 
                                                      and on_bnd),
                                   bc_type = 'Outlet',
                                   func = {'p': Constant(0.0)},
                                   mf = self.mf)
                    
        # Now that the boundaries are marked we can create the bump by 
        # squeezing the mesh
        x[:, 1] = x[:, 1]*(2. - self.bump(x[:, 0]))/2. + self.bump(x[:, 0])
        return m, [walls, inlet, outlet]
                        
    def __info__(self):
        return 'Adverse pressure gradient channel flow with a bump on the lower wall'

if __name__ == '__main__':
    import cbc.cfd.icns as icns
    from cbc.cfd.icns import solver_parameters
    set_log_active(False)
    problem_parameters['time_integration']='Transient'
    problem_parameters['Nx'] = 80
    problem_parameters['Ny'] = 60
    problem_parameters['T'] = 10.
    problem_parameters['max_iter'] = 1
    #problem_parameters['pressure_bc'] = True
    problem_parameters['plot_velocity'] = False
    solver_parameters = recursive_update(solver_parameters, 
    dict(degree=dict(u=1),
         pdesubsystem=dict(u=101, p=101, velocity_update=101, up=1), 
         linear_solver=dict(u='bicgstab', p='gmres', velocity_update='bicgstab'), 
         precond=dict(u='ilu', p='amg', velocity_update='ilu'),
         iteration_type='Picard', max_iter=5)
         )
    NS_problem = apbl(problem_parameters)
    NS_solver = icns.NSFullySegregated(NS_problem, solver_parameters)        
    #NS_solver = icns.NSCoupled(NS_problem, solver_parameters) 
    
    NS_solver.pdesubsystems['u0'].prm['monitor_convergence'] = True
    NS_problem.solve()
    