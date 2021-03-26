__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2011-08-22"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU GPL version 3 or any later version"

from NSProblem import *
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from numpy import array, zeros, floor

MCAtime = array([    0.,    27.,    42.,    58.,    69.,    88.,   110.,   130.,                                                                    
        136.,   168.,   201.,   254.,   274.,   290.,   312.,   325.,                                                                                      
        347.,   365.,   402.,   425.,   440.,   491.,   546.,   618.,                                                                                      
        703.,   758.,   828.,   897.,  1002.])
    
MCAval = array([ 390.        ,  398.76132931,  512.65861027,  642.32628399,                                                        
        710.66465257,  770.24169184,  779.00302115,  817.55287009,                                                                                          
        877.12990937,  941.96374622,  970.        ,  961.2386707 ,                                                                                          
        910.42296073,  870.12084592,  843.83685801,  794.7734139 ,                                                                                          
        694.89425982,  714.16918429,  682.62839879,  644.07854985,                                                                                          
        647.58308157,  589.75830816,  559.96978852,  516.16314199,                                                                                          
        486.37462236,  474.10876133,  456.58610272,  432.05438066,  390.]
        )*0.001*2./3.

class aneurysm(NSProblem):
    
    def __init__(self, parameters):
        NSProblem.__init__(self, parameters=parameters)
        #self.mesh = Mesh("../data/100_1314k.xml.gz")
        self.mesh = Mesh("../data/Aneurysm.xml.gz")
        self.boundaries = self.create_boundaries()
        
        # To initialize solution set the dictionary q0: 
        #self.q0 = Initdict(u = ('0', '0', '0'), p = ('0')) # Or not, zero is default anyway
        
    def create_boundaries(self):
        # Define the spline for the heart beat
        self.inflow_t_spline = ius(MCAtime, MCAval)
        
        # Preassemble normal vector on inlet
        n = self.n = FacetNormal(self.mesh)        
        self.normal = [assemble(-n[i]*ds(2), mesh=self.mesh) for i in range(3)]
        
        # Area of inlet 
        self.A0 = assemble(Constant(1.)*ds(2), mesh=self.mesh)
        
        # Create dictionary used for Dirichlet inlet conditions. Values are assigned in prepare, called at the start of a new timestep                       
        self.inflow = {'u' : Constant((0, 0, 0)),
                       'u0': Constant(0),
                       'u1': Constant(0),
                       'u2': Constant(0)}

        # Pressures on outlets are specified by DirichletBCs, values are computed in prepare
        self.p_out1 = Constant(0)
        self.p_out2 = Constant(0)

        # Specify the boundary subdomains and hook up dictionaries for DirichletBCs
        walls     = MeshSubDomain(0, 'Wall')
        inlet     = MeshSubDomain(2, 'VelocityInlet', self.inflow)
        pressure1 = MeshSubDomain(1, 'ConstantPressure', {'p': self.p_out1})
        pressure2 = MeshSubDomain(3, 'ConstantPressure', {'p': self.p_out2})
        
        return [walls, inlet, pressure1, pressure2]
        
    def prepare(self):
        """Called at start of a new timestep. Set the outlet pressure at new time."""
        solver = self.pdesystems['Navier-Stokes']
        t = self.t - floor(self.t/1002.0)*1002.0
        u_mean = self.inflow_t_spline(t)[0]/self.A0        
        self.inflow['u'].assign(Constant(u_mean*array(self.normal)))
        for i in range(3):
            self.inflow['u'+str(i)].assign(u_mean*self.normal[i])
            
        info_green('UMEAN = {0:2.5f} at time {1:2.5f}'.format(u_mean, self.t))
        # First time around we assemble some vectors that can be used to compute 
        # the outlet pressures with merely inner products and no further assembling.
        if not hasattr(self, 'A1'): 
            self.A1 = []
            self.A3 = []
            v = solver.vt['u0']
            for i in range(3):
                self.A1.append(assemble(v*self.n[i]*ds(1)))
                self.A3.append(assemble(v*self.n[i]*ds(3)))
        
        # Compute outlet pressures fast
        p1 = 0
        p2 = 0
        for i in range(3):
            p1 += self.A1[i].inner(solver.u_[i].vector())
            p2 += self.A3[i].inner(solver.u_[i].vector())
        self.p_out1.assign(p1)
        self.p_out2.assign(p2)            
        # Or the slow approach:
        #self.p_out1.assign(assemble(dot(solver.u_, self.n)*ds(1)))
        #self.p_out2.assign(assemble(dot(solver.u_, self.n)*ds(3)))
        
        info_green('Pressure outlet 1 = {0:2.5f}'.format(self.p_out1(0)))
        info_green('Pressure outlet 3 = {0:2.5f}'.format(self.p_out2(0)))
        
        if self.tstep % 10 == 0:
            print 'Memory usage = ', self.getMyMemoryUsage()

if __name__ == '__main__':
    from cbc.cfd.icns import NSFullySegregated, NSSegregated, solver_parameters
    import time
    parameters["linear_algebra_backend"] = "PETSc"
    set_log_active(True)
    problem_parameters['viscosity'] = 0.00345
    problem_parameters['T'] = 0.05
    problem_parameters['dt'] = 0.05
    problem_parameters['iter_first_timestep'] = 2
    solver_parameters = recursive_update(solver_parameters, 
    dict(degree=dict(u=1,u0=1,u1=1,u2=1),
         pdesubsystem=dict(u=101, p=101, velocity_update=101), 
         linear_solver=dict(u='bicgstab', p='gmres', velocity_update='bicgstab'), 
         precond=dict(u='jacobi', p='hypre_amg', velocity_update='jacobi'))
         )
    
    problem = aneurysm(problem_parameters)
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
