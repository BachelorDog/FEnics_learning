__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-08-30"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"
"""
Base class for all Navier-Stokes problems.
"""
from cbc.pdesys.Problem import *

problem_parameters = copy.deepcopy(default_problem_parameters)
problem_parameters = recursive_update(problem_parameters, {
    'Re': 50.,
    'viscosity': None,
    'cfl': 1.0,
    'save_solution': False,
    'plot_velocity': False,
    'plot_pressure': False,    
    'turbulence_model': 'Laminar',
})

class NSProblem(Problem):
    
    def __init__(self, mesh=None, parameters=None):
        Problem.__init__(self, mesh=mesh, parameters=parameters)
        try:
            self.output_location = os.environ['CBCRANS']
        except KeyError:
            info_red('Set the environment variable CBCRANS to control the location of stored results')            
            self.output_location = os.getcwd()
        
    def body_force(self):
        return Constant((0.,)*self.mesh.geometry().dim())
        
    def timestep(self, U=1.):
        h = self.mesh.hmin()
        dt = self.prm['cfl']*h**2/(U*(self.prm['viscosity'] + h*U))
        n  = int(self.prm['T'] / dt + 1.0)
        return self.prm['T']/n
        
    def update(self):
        """Called at end of timestep for transient simulations or at the end of
        iterations for steady simulations."""
        if self.prm['save_solution']:
            if self.prm['time_integration'] == 'Steady':
                i = self.total_number_iters
            else:
                i = self.tstep
            result_path = os.path.join(self.output_location, 'cbc', 
                                       'rans', 'results')
            for pt in (self.__class__.__name__,
                       self.NS_solver__class__.__name__, 
                       self.prm['Model']):
                result_path = self.result_path = os.path.join(result_path, pt)
                if not os.path.exists(result_path):
                    os.mkdir(result_path)         

            if (i - 1) % self.prm["save_solution"] == 0:
                # Create files for saving
                for name in self.system_names:
                    if not name in self.resultfile:
                        self.resultfile[name] = File(os.path.join(result_path, name + ".pvd"))
                    self.resultfile[name] << self.q_[name]
                
        if self.prm['plot_velocity']:
            if isinstance(self.NS_solver.u_, ufl.tensors.ListTensor):
                V = VectorFunctionSpace(self.mesh, 'CG', 
                                        self.NS_solver.prm['degree']['u0'])
                u_ = project(self.NS_solver.u_, V)
                if hasattr(self, 'u_plot'):
                    self.u_plot.vector()[:] = u_.vector()[:]
                else:
                    self.u_plot = u_
            else:
                self.u_plot = self.NS_solver.u_
            plot(self.u_plot, title='Velocity', rescale=True)

        if self.prm['plot_pressure']: plot(self.NS_solver.p_, title='Pressure',
                                           rescale=True)
