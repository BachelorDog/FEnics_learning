__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-11-03"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"
""" 
    Taylor Green test problem in 3D
"""
from NSProblem import *

# Specify the initial velocity field
initial_velocity = Initdict( 
    u=('2./sqrt(3.)*sin(2.*pi/3.)*sin(x[0])*cos(x[1])*cos(x[2])',
       '2./sqrt(3.)*sin(-2.*pi/3.)*cos(x[0])*sin(x[1])*cos(x[2])',
       '0.0'),
    p=('0'))

# Required parameters
problem_parameters['Nx'] = 16
problem_parameters['Ny'] = 16
problem_parameters['Nz'] = 16
problem_parameters['Re'] = 100.

# Problem definition
class TaylorGreen(NSProblem):

    def __init__(self, parameters):
        NSProblem.__init__(self, parameters=parameters)
        self.mesh = self.gen_mesh()
        # Set viscosity
        self.prm['viscosity'] = 1.0/self.prm['Re']
        self.prm['dt'] = self.timestep()    
        self.boundaries = self.create_boundaries()
        self.vorticity_file = File("vorticity.pvd")
        self.q0 = initial_velocity
    
    def gen_mesh(self):
        m = UnitCube(self.prm['Nx'], self.prm['Ny'], self.prm['Nz'])
        scale = 2*(m.coordinates() - 0.5)*pi
        m.coordinates()[:, :] = scale
        return m
                
    def create_boundaries(self):
        
        self.mf = FacetFunction("uint", self.mesh) # Facets
        self.mf.set_all(0)
        
        def periodic_mapX(x, y):
            y[0] = x[0] - 2.0*DOLFIN_PI
            y[1] = x[1]
            y[2] = x[2]
            
        def periodic_mapY(x, y):
            y[0] = x[0] 
            y[1] = x[1] - 2.0*DOLFIN_PI
            y[2] = x[2]
            
        def periodic_mapZ(x, y):
            y[0] = x[0] 
            y[1] = x[1]
            y[2] = x[2] - 2.0*DOLFIN_PI
                
        periodicX = FlowSubDomain(lambda x, on_boundary: near(x[0], -DOLFIN_PI) and on_boundary,
                                bc_type = 'Periodic',
                                mf = self.mf,
                                periodic_map = periodic_mapX)

        periodicY = FlowSubDomain(lambda x, on_boundary: near(x[1], -DOLFIN_PI) and on_boundary,
                                bc_type = 'Periodic',
                                mf = self.mf,
                                periodic_map = periodic_mapY)
                                
        periodicZ = FlowSubDomain(lambda x, on_boundary: near(x[2], -DOLFIN_PI) and on_boundary,
                                bc_type = 'Periodic',
                                mf = self.mf,
                                periodic_map = periodic_mapZ)
                                
        return [periodicX, periodicY, periodicZ]
       
    def update(self):
        if (self.tstep-1) % self.NS_solver.prm['save_solution'] == 0:
            V = MixedFunctionSpace([self.NS_solver.V['u0']]*3)
            ff = project(curl(self.NS_solver.u_), V)
            self.vorticity_file << ff
            
    def info(self):
        return "Taylor-Green vortex"

if __name__ == '__main__':
    from cbc.cfd import icns                    # Navier-Stokes solvers
    from cbc.cfd.icns import solver_parameters  # parameters to NS solver
    set_log_active(True)
    problem_parameters['time_integration']='Transient'
    problem_parameters['Nx'] = 24
    problem_parameters['Ny'] = 24
    problem_parameters['Nz'] = 24
    problem_parameters['T'] = 1.
    solver_parameters = recursive_update(solver_parameters, 
    dict(degree=dict(u=1),
         pdesubsystem=dict(u=101, p=101, velocity_update=101, up=1), 
         linear_solver=dict(u='bicgstab', p='gmres', velocity_update='bicgstab'), 
         precond=dict(u='jacobi', p='ilu', velocity_update='jacobi'),
         save_solution=10)
         )
    
    NS_problem = TaylorGreen(problem_parameters)
    NS_solver = icns.NSFullySegregated(NS_problem, solver_parameters)
    #NS_solver = icns.NSSegregated(NS_problem, solver_parameters)
    #NS_solver = icns.NSCoupled(NS_problem, solver_parameters)
    
    t0 = time()
    NS_problem.solve()
    print 'time = ', time()-t0
    print summary()
    plot(NS_solver.u_)
    interactive()
    