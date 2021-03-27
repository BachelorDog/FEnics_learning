__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-11-03"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"
""" 
Beltrami test problem in 3D
"""
from NSProblem import *

# Specify the initial velocity field

u_params = {'a': pi/4.0, 'd': pi/2.0, 'E': e,             'etabyrho': 1.0, 't': 0.0}
p_params = {'a': pi/4.0, 'd': pi/2.0, 'E': e, 'rho': 1.0, 'etabyrho': 1.0, 't': 0.0}

exact = dict( 
    u=('-(({a}*(pow({E},{a}*x[2])*cos({a}*x[0] + {d}*x[1]) + pow({E},{a}*x[0])*sin({a}*x[1] +  {d}*x[2])))/pow({E},pow({d},2)*{t}*{etabyrho}))',
       '-(({a}*(pow({E},{a}*x[0])*cos({a}*x[1] + {d}*x[2]) + pow({E},{a}*x[1])*sin({d}*x[0] + {a}*x[2])))/pow({E},pow({d},2)*{t}*{etabyrho}))',
       '-(({a}*(pow({E},{a}*x[1])*cos({d}*x[0] + {a}*x[2]) + pow({E},{a}*x[2])*sin({a}*x[0] + {d}*x[1])))/pow({E},pow({d},2)*{t}*{etabyrho}))'),
    p=('-({rho}/2.0)*(pow({a},2)*(pow({E},2*{a}*x[0]) + pow({E},2*{a}*x[1]) + pow({E},2*{a}*x[2]) + 2*pow({E},{a}*(x[1] + x[2]))*cos({d}*x[0] + {a}*x[2])*sin({a}*x[0] + {d}*x[1]) + 2*pow({E},{a}*(x[0] + x[1]))*cos({a}*x[1] + {d}*x[2])*sin({d}*x[0] + {a}*x[2]) + 2*pow({E},{a}*(x[0] + x[2]))*cos({a}*x[0] + {d}*x[1])*sin({a}*x[1] + {d}*x[2])))/(pow({E},pow({d},2)*{t}*{etabyrho}))'))


# Define dictionary for setting initial velocity and pressure
initial_velocity = Initdict(
    u = (exact['u'][0].format(**u_params),
         exact['u'][1].format(**u_params),
         exact['u'][2].format(**u_params)),
    p =  exact['p'].format(**p_params)
    )

# Define dictionary for analytical solution used as boundary condition on velocity
u_params['t'] = 't'
p_params['t'] = 't'
exact_velocity = dict(
    u = Expression((exact['u'][0].format(**u_params),
                    exact['u'][1].format(**u_params),
                    exact['u'][2].format(**u_params)),  t=0),
    u0 = Expression((exact['u'][0].format(**u_params)), t=0),
    u1 = Expression((exact['u'][1].format(**u_params)), t=0),
    u2 = Expression((exact['u'][2].format(**u_params)), t=0),
    )

# Problem definition
class Beltrami(NSProblem):

    def __init__(self, parameters):
        NSProblem.__init__(self, parameters=parameters)
        self.mesh = self.gen_mesh()
        # Set viscosity
        self.prm['viscosity'] = 1.0/self.prm['Re']
        self.prm['dt'] = self.prm['T']/int(self.prm['T']/0.2/self.mesh.hmin() + 1.0)
        #self.prm['dt'] = self.prm['T']/ceil(self.prm['T']/0.25/self.mesh.hmin())
        self.boundaries = self.create_boundaries()
        self.q0 = initial_velocity
    
    def gen_mesh(self):
        m = UnitCube(self.prm['Nx'], self.prm['Ny'], self.prm['Nz'])
        scale = 2*(m.coordinates() - 0.5)
        m.coordinates()[:, :] = scale
        return m
                
    def create_boundaries(self):
        
        domain = FlowSubDomain(lambda x, on_bnd: on_bnd,
                               bc_type = 'VelocityInlet', # Dirichlet on u, nothing on p
                               func = exact_velocity)
                                
        return [domain]
        
    def prepare(self):
        for val in exact_velocity.itervalues():
            val.t = self.t        
        
    def update(self):
        if self.tstep % 10 == 0:
            print 'Memory usage = ', self.getMyMemoryUsage()
        #self.functional()
        
    def functional(self):
        # errornorm doesn't work with the ListTensor u_ of the segregated solver
        if hasattr(self.NS_solver, 'u0_'):
            f  = sqr(errornorm(exact_velocity['u0'], self.NS_solver.u0_)/norm(exact_velocity['u0'], mesh=self.mesh))
            f += sqr(errornorm(exact_velocity['u1'], self.NS_solver.u1_)/norm(exact_velocity['u1'], mesh=self.mesh))
            f += sqr(errornorm(exact_velocity['u2'], self.NS_solver.u2_)/norm(exact_velocity['u2'], mesh=self.mesh))
        else:
            f = errornorm(self.NS_solver.u_, exact_velocity['u'])
        error = sqrt(f/3.)
        info_red('Errornorm = {0:2.5e}'.format(error))
        return error
                   
    def info(self):
        return "Beltrami flow"

if __name__ == '__main__':
    from cbc.cfd import icns                    # Navier-Stokes solvers
    from cbc.cfd.icns import solver_parameters  # parameters to NS solver
    import time
    import sys
    parameters["linear_algebra_backend"] = "PETSc"
    #set_log_active(True)
    #set_log_level(5)
    
    mesh_sizes = [5, 8, 11, 16, 23, 32, 64]
    try:
        N = eval(sys.argv[-1])
    except:
        N = 2    
    problem_parameters['time_integration']='Transient'
    problem_parameters['Nx'] = mesh_sizes[N]
    problem_parameters['Ny'] = mesh_sizes[N]
    problem_parameters['Nz'] = mesh_sizes[N]
    problem_parameters['Re'] = 1.
    problem_parameters['T'] = 0.5
    solver_parameters = recursive_update(solver_parameters, 
    dict(degree=dict(u=2, u0=1, u1=1, u2=1),
         pdesubsystem=dict(u=101, p=101, velocity_update=101, up=1), 
         linear_solver=dict(u='bicgstab', p='gmres', velocity_update='bicgstab'), 
         precond=dict(u='jacobi', p='hypre_amg', velocity_update='ilu'))
         )
    
    problem = Beltrami(problem_parameters)
    solver = icns.NSFullySegregated(problem, solver_parameters)
    #solver = icns.NSSegregated(problem, solver_parameters)
    #solver = icns.NSCoupled(problem, solver_parameters)
    
    for name in ['u0', 'u1', 'u2', 'p', 'u0_update', 'u1_update', 'u2_update']:
        solver.pdesubsystems[name].prm['monitor_convergence'] = True
        solver.pdesubsystems[name].linear_solver.parameters['relative_tolerance'] = 1e-12
        solver.pdesubsystems[name].linear_solver.parameters['absolute_tolerance'] = 1e-25
    
    t0 = time.time()
    problem.solve()
    error = problem.functional()
    t1 = time.time()-t0
    print 'time = ', t1
    print list_timings()
    
    dump_result(problem, solver, t1, error)
    
    #plot(solver.u_)
    #interactive()
    
