__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2011-12-02"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU GPL version 3 or any later version"

from NSProblem import *
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from numpy import array, zeros, floor
from aneurysm import MCAtime, MCAval

class Challenge(NSProblem):
    
    def __init__(self, parameters):
        NSProblem.__init__(self, parameters=parameters)
        self.mesh = Mesh("../data/mesh_coarse.xml")
        self.boundaries = self.create_boundaries()
        
        # To initialize solution set the dictionary q0: 
        #self.q0 = Initdict(u = ('0', '0', '0'), p = ('0')) # Or not, zero is default anyway
        
    def create_boundaries(self):
        # Define the spline for the heart beat
        self.inflow_t_spline = ius(MCAtime, MCAval)
        
        # Preassemble normal vector on inlet
        n = self.n = FacetNormal(self.mesh)        
        self.normal = [assemble(-n[i]*ds(4), mesh=self.mesh) for i in range(3)]
        
        # Area of inlet 
        self.A0 = assemble(Constant(1.)*ds(4), mesh=self.mesh)
        
        # Create dictionary used for Dirichlet inlet conditions. Values are assigned in prepare
        self.inflow = {'u' : Constant((0, 0, 0)),
                       'u0': Constant(0),
                       'u1': Constant(0),
                       'u2': Constant(0)}

        # Pressures on outlets are specified by DirichletBCs
        self.p_out1 = Constant(0)

        # Specify the boundary subdomains and hook up dictionaries for DirichletBCs
        walls     = MeshSubDomain(3, 'Wall')
        inlet     = MeshSubDomain(4, 'VelocityInlet', self.inflow)
        pressure1 = MeshSubDomain(5, 'ConstantPressure', {'p': self.p_out1})
        
        return [walls, inlet, pressure1]
        
    def prepare(self):
        """Called at start of a new timestep."""
        t = self.t - floor(self.t/1002.0)*1002.0
        u_mean = self.inflow_t_spline(t)[0]/self.A0        
        self.inflow['u'].assign(Constant(u_mean*array(self.normal)))
        for i in range(3):
            self.inflow['u'+str(i)].assign(u_mean*self.normal[i])

if __name__ == '__main__':
    from cbc.cfd.icns import NSFullySegregated, NSSegregated, solver_parameters
    import time
    parameters["linear_algebra_backend"] = "Epetra"
    set_log_active(True)
    problem_parameters['viscosity'] = 0.00345
    problem_parameters['T'] = 0.5
    problem_parameters['dt'] = 0.05
    problem_parameters['iter_first_timestep'] = 2
    solver_parameters = recursive_update(solver_parameters, 
    dict(degree=dict(u=1,u0=1,u1=1,u2=1),
         pdesubsystem=dict(u=101, p=101, velocity_update=101), 
         linear_solver=dict(u='bicgstab', p='gmres', velocity_update='bicgstab'), 
         precond=dict(u='jacobi', p='amg', velocity_update='jacobi'))
         )
    
    problem = Challenge(problem_parameters)
    solver = NSFullySegregated(problem, solver_parameters)
    for name in solver.system_names:
        solver.pdesubsystems[name].prm['monitor_convergence'] = False
    #solver.pdesubsystems['u0_update'].prm['monitor_convergence'] = True
    #solver.pdesubsystems['u1_update'].prm['monitor_convergence'] = True
    #solver.pdesubsystems['u2_update'].prm['monitor_convergence'] = True
    t0 = time.time()
    problem.solve()
    t1 = time.time() - t0

    # Save solution
    #V = VectorFunctionSpace(problem.mesh, 'CG', 1)
    #u_ = project(solver.u_, V)
    #file1 = File('u.pvd')
    #file1 << u_

    print list_timings()

    dump_result(problem, solver, t1, 0)
