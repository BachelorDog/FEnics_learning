__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2011-11-30"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU GPL version 3 or any later version"

from NSProblem import *
                        
class CSF(NSProblem):
    
    def __init__(self, parameters):
        NSProblem.__init__(self, parameters=parameters)
        self.mesh = Mesh('../data/patient5_small.xml')
        self.boundaries = self.create_boundaries()
        # To initialize solution set the dictionary q0: 
        #self.q0 = Initdict(u = ('0', '0', '0'), p = ('0')) # Or not, zero is default anyway
        
    def create_boundaries(self):
        # Pressures are specified on top and bottom, values can be modified in prepare
        self.p_top = Constant(0)
        self.p_bottom = Constant(1)

        # Specify the boundary subdomains and hook up dictionaries for DirichletBCs
        top         = MeshSubDomain(11, 'ConstantPressure', {'p': self.p_top})
        bottom      = MeshSubDomain(12, 'ConstantPressure', {'p': self.p_bottom})
        outer_walls = MeshSubDomain(13, 'Wall')
        inner_walls = MeshSubDomain(14, 'Wall')
        
        return [inner_walls, outer_walls, top, bottom]
        
    def prepare(self):
        """Called at start of a new timestep. Set new pressure BCs at new time."""
        #self.p_top.assign(1)
        #self.p_bottom.assign(0)
        info_green('Pressure Top    = {0:2.5f}'.format(self.p_top(0)))
        info_green('Pressure Bottom = {0:2.5f}'.format(self.p_bottom(0)))

if __name__ == '__main__':
    from cbc.cfd.icns import NSFullySegregated, NSSegregated, solver_parameters
    import time
    parameters["linear_algebra_backend"] = "PETSc"
    set_log_active(True)
    problem_parameters['viscosity'] = 0.001
    problem_parameters['T'] = 1.
    problem_parameters['dt'] = 0.01
    problem_parameters['iter_first_timestep'] = 2
    solver_parameters = recursive_update(solver_parameters, 
    dict(degree=dict(u=1,u0=1,u1=1,u2=1),
         pdesubsystem=dict(u=101, p=101, velocity_update=101), 
         linear_solver=dict(u='bicgstab', p='gmres', velocity_update='bicgstab'), 
         precond=dict(u='jacobi', p='hypre_amg', velocity_update='jacobi'))
         )
    
    problem = CSF(problem_parameters)
    solver = NSFullySegregated(problem, solver_parameters)
    t0 = time.time()
    problem.solve()
    t1 = time.time() - t0

    #V = VectorFunctionSpace(problem.mesh, 'CG', 1)
    #u_ = project(solver.u_, V)
    #file1 = File('u.pvd')
    #file1 << u_

    print list_timings()

    dump_result(problem, solver, t1, 0)
    
