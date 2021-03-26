__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-10-28"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"

from NSProblem import *

class Submesh(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 0.25 - DOLFIN_EPS and x[1] > 0.25 - DOLFIN_EPS

# Problem definition
class Lshape(NSProblem):

    def __init__(self, parameters):
        NSProblem.__init__(self, parameters=parameters)
        
        # Create the L-shaped mesh 
        mesh_ = UnitSquare(self.prm['Nx'], self.prm['Ny'])
        subm = Submesh()
        self.mf1 = MeshFunction("uint", mesh_, 2)
        self.mf1.set_all(0)
        subm.mark(self.mf1, 1)
        self.mesh = mesh = SubMesh(mesh_, self.mf1, 0)
        
        # Create meshfunction to mark all boundaries
        self.mf = FacetFunction("uint", mesh)
        self.mf.set_all(0)
        
        self.boundaries = self.create_boundaries()
        
        # Set some constants
        self.nu = self.prm['viscosity'] = 1./self.prm['Re']
        
        self.prm['dt'] = self.timestep()

        # dictionary q0 is used to initialize the flow
        transient = self.prm['time_integration']=='Transient'
        transient_q0 = Initdict({'p': '0', 'u': ('0', '0')})
        steady_q0 = Initdict({'p': 'x[1]', 'u': ('0', '0')})
        self.q0 = transient_q0 if transient else steady_q0

    def create_boundaries(self):
        
        # Create pressure function for inlet
        self.p_in = Expression("sin(pi*t)", t=0.)
        if self.prm['time_integration'] == 'Steady': 
            self.p_in.t = 0.5
                
        # Create list of boundaries and mark them in the meshfunction mf
        inlet  = FlowSubDomain(lambda x, on_boundary: near(x[1] - 1., 0.) and on_boundary,
                              bc_type='ConstantPressure',
                              func = {'p': self.p_in},     # Specify pressure on inlet 
                              mf = self.mf)                # Meshfunction identifying boundaries

        outlet = FlowSubDomain(lambda x, on_boundary: near(x[0] - 1., 0.) and on_boundary,
                              bc_type = 'ConstantPressure',
                              func = {'p': Constant(0.0)},
                              mf = self.mf)

        walls = FlowSubDomain(lambda x, on_boundary: (near(x[0], 0.) or near(x[1], 0.) or
                                                     (x[0] > 0.25 - 5*DOLFIN_EPS  and 
                                                      x[1] > 0.25 - 5*DOLFIN_EPS) and
                                                      on_boundary),
                             bc_type = 'Wall',
                             mf = self.mf)
                             
        return [inlet, walls, outlet]
                                                                
    def prepare(self):
        """Called at start of a new timestep. Set the pressure at new time."""
        if self.prm['time_integration'] == 'Transient':
            self.p_in.t = self.t
        
    def __info__(self):
        return "Time varying L-shaped domain"

if __name__ == '__main__':
    import cbc.cfd.icns as icns
    from cbc.cfd.icns import solver_parameters
    import time
    set_log_active(False)
    problem_parameters['Nx'] = 40
    problem_parameters['Ny'] = 40
    problem_parameters['T'] = 10.
    problem_parameters['time_integration']='Steady'
    problem_parameters['max_iter'] = 10
    problem_parameters['plot_velocity'] = True
    solver_parameters = recursive_update(solver_parameters, 
    dict(degree=dict(u=2, u0=1, u1=1),
         pdesubsystem=dict(u=101, p=101, velocity_update=101, up=1), 
         linear_solver=dict(u='bicgstab', p='gmres', velocity_update='bicgstab', up='lu'), 
         precond=dict(u='jacobi', p='amg', velocity_update='ilu'),
         iteration_type='Newton')
         )
    problem = Lshape(problem_parameters)
    #solver = icns.NSFullySegregated(problem, solver_parameters)
    solver = icns.NSCoupled(problem, solver_parameters)    
    t0 = time.time()
    problem.solve()
    t1 = time.time() - t0
    
    #dump_result(problem, solver, t1, 0)
