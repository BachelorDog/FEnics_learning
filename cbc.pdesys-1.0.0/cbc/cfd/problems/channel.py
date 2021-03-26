__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-08-30"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"

""" 
Set up laminar channel flow test case. 
The channel spans  -1 <= y <= 1.

For laminar flows the flow is usually caracterized by Re based on the 
maximum velocity, i.e. Re = U/nu. If the steady solution is set to 
u = 1-y**2, then Re=1/nu and the pressure gradient is 2/Re. This 
is based on an exact integral across the channel height:

\int_{-1}^{1}(dp/dx) dV = -\int_{-1}^{1} nu*d^2/dy^2u dV 
2*dp/dx = - nu*(du/dy_top- du/dy_bottom) = -1/Re*(-2 - 2)
dp/dx = 2/Re
    
"""    

from NSProblem import *
from numpy import arctan, array

# Laminar velocity profiles that can be used in Expressions
# for initializing the solution or prescribing a VelocityInlet.
laminar_velocity = Initdict(u=(("(1.-x[1]*x[1])",  "0")), 
                            p=("0"))

zero_velocity = Initdict(u=(("0",  "0")), 
                         p=("0"))
#2./8.*(1.-x[0])
# Default parameters for channel
problem_parameters['periodic'] = False
problem_parameters['Nx'] = 16
problem_parameters['Ny'] = 16
problem_parameters['L'] = 1.

class channel(NSProblem):
    def __init__(self, parameters):
        NSProblem.__init__(self, parameters=parameters)
        self.mesh = self.gen_mesh()
        self.prm['viscosity'] = 1./self.prm['Re']
        self.prm['dt'] = self.timestep()
        self.boundaries = self.create_boundaries()
        
        # The GRPC solver, which uses pressure correction, requires the correct pressure to be set initially
        #zero_velocity['p'] = Expression('2./Re*L*(1.-x[0])', Re=self.prm['Re'], L=self.L)
        
        transient = self.prm['time_integration']=='Transient'
        self.q0 = zero_velocity if transient else laminar_velocity
        
    def gen_mesh(self):
        self.L = problem_parameters['L']
        m = Rectangle(0., -1., self.L, 1., self.prm['Nx'], self.prm['Ny'])
        # Create stretched mesh in y-direction
        x = m.coordinates()        
        x[:, 1] = arctan(pi*(x[:, 1]))/arctan(pi) 
        #x[:, 1] = 0.5*x[:, 1]
        return m        
        
    def create_boundaries(self):
        self.mf = FacetFunction("uint", self.mesh) # Facets
        self.mf.set_all(0)
                        
        walls = FlowSubDomain(lambda x, on_boundary: ((near(x[1], -1.) or 
                                                       near(x[1], 1.)) and
                                                       on_boundary),
                              bc_type = 'Wall',
                              mf = self.mf)

        #symmetry = FlowSubDomain(lambda x, on_boundary: near(x[1], 0.) and on_boundary,
                                #bc_type = 'Symmetry',
                                #mf = self.mf)

        if self.prm['periodic']:
            
            self.pressure_gradient = Constant((2./self.prm['Re'], 0.))

            def periodic_map(x, y):
                y[0] = x[0] - self.L
                y[1] = x[1]
            p = periodic_map
            p.L = self.L
                
            periodic = FlowSubDomain(lambda x, on_boundary: abs(x[0]) < 10.*DOLFIN_EPS and on_boundary,
                                    bc_type = 'Periodic',
                                    mf = self.mf,
                                    periodic_map = p)
            bcs = [walls, periodic]
            
        else:
            
            self.pressure_gradient = Constant((0, 0))
            
            # Note that VelocityInlet should only be used for Steady problem
            #inlet    = FlowSubDomain(lambda x, on_boundary: near(x[0], 0.) and on_boundary,
                                    #bc_type = 'VelocityInlet',
                                    #func = laminar_velocity,
                                    #mf = self.mf)
            #outlet   = FlowSubDomain(lambda x, on_boundary: near(x[0], self.L) and on_boundary,
                                    #bc_type = 'Outlet',
                                    #func = {'p': Constant(0.0)},
                                    #mf = self.mf)
                                    
            inlet    = FlowSubDomain(lambda x, on_boundary: near(x[0], 0.) and on_boundary,
                                    bc_type = 'ConstantPressure',
                                    func = {'p': Constant(2./self.prm['Re']*self.L)},
                                    mf = self.mf)
                                
            outlet   = FlowSubDomain(lambda x, on_boundary: near(x[0], self.L) and on_boundary,
                                    bc_type = 'ConstantPressure',
                                    func = {'p': Constant(0.0)},
                                    mf = self.mf)
                                
            bcs = [walls, inlet, outlet]                    
        
        return bcs
                
    def body_force(self):
        return self.pressure_gradient
        
    def functional(self, u):
        x = array((1.0, 0.))
        values = array((0.0, 0.0))
        u.eval(values, x)
        return values[0]

    def reference(self):
        num_terms = 10000
        u = 1.0
        c = 1.0
        for n in range(1, 2*num_terms, 2):
            a = 32. / (pi**3*n**3)
            b = (0.25/self.prm['Re'])*pi**2*n**2
            c = -c
            u += a*exp(-b*self.t)*c
        return u
        
    def error(self):
        return self.functional(self.NS_solver.u_) - self.reference()
        
    def __info__(self):
        return 'Periodic channel flow'

if __name__ == '__main__':
    # Solve laminar channel problem
    from cbc.cfd import icns                    # Navier-Stokes solvers
    from cbc.cfd.icns import solver_parameters  # parameters to NS solver
    set_log_active(True)
    problem_parameters['time_integration'] = 'Transient'
    problem_parameters['T'] = 0.5
    problem_parameters['Re'] = 8.
    problem_parameters['max_iter'] = 1
    problem_parameters['max_err'] = 1e-10
    problem_parameters['plot_velocity'] = False # plot velocity at end of timestep
    problem_parameters['periodic'] = True      # Use or not periodic boundary conditions
    
    solver_parameters = recursive_update(solver_parameters, 
    dict(degree=dict(u=2),
         pdesubsystem=dict(u=30, p=30, velocity_update=0, up=1), max_iter=5,         # GRPC
         linear_solver=dict(u='lu', p='lu', velocity_update='bicgstab'), 
         precond=dict(u='jacobi', p='amg', velocity_update='jacobi'))
    )
    
    # Set up problem
    NS_problem = channel(problem_parameters)
    
    # Choose Navier-Stokes solver
    #NS_solver = icns.NSFullySegregated(NS_problem, solver_parameters)
    NS_solver = icns.NSSegregated(NS_problem, solver_parameters)
    #NS_solver = icns.NSCoupled(NS_problem, solver_parameters)
    
    # Solve the problem
    NS_problem.solve()
    plot(NS_solver.u_)

    # Check where the time went
    print list_timings()
    