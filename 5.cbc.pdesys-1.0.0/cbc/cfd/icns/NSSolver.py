__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-08-30"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"
"""
Base class for all Navier-Stokes solvers.
"""
from cbc.pdesys.PDESystem import *

solver_parameters  = copy.deepcopy(default_solver_parameters)
solver_parameters = recursive_update(solver_parameters, {
    'convection_form': 'Standard',
    'stabilization_prm': 0.01,
    'space': dict(u=VectorFunctionSpace),
    'degree': dict(u=2),
    'apply': defaultdict(lambda: 'use_formula'),
    'familyname': 'Navier-Stokes',
    'plot_velocity': False,
    'plot_pressure': False
})

# For segregated solvers we can overload Subdict to use 'u' options where 
# 'u1', 'u2' or 'u3' are missing
# Not neccesary, but simplifies the problem interface (we don't have to
# specify options for all individual velocity components)
def __missing__(self, key):
    """Subdict is a dictionary that first looks for missing keys in the
    solver_namespace through sub_name. 
    Here we overload this method for the segregated velocities 'u0', 'u1'
    and 'u2' in such a way that we use options for 'u' as default.
    
    We will here overload such that the prm dictionary looks for keys as:
       
       1) prm[key]['u1']
       2) prm[key]['u']
       3  prm[key]
    
    If we set
        solver_namespace['prm']['linear_solver']['u'] = 'bicgstab'
    then this option will be used unless we also specify
        solver_namespace['prm']['linear_solver']['u1'] = 'gmres'
        
    Some parameters do not use sub_name as keys and these will fall back on 3)
    """
    if self.sub_name in ('u0', 'u1', 'u2'):
        try:
            if self.sub_name in self.solver_namespace['prm'][key]:
                self[key] = self.solver_namespace['prm'][key][self.sub_name]
            else:
                self[key] = self.solver_namespace['prm'][key]['u']
            info_green("Adding ['{0:s}']['{1:s}'] = {2:s} to pdesubsystem {3:s}".format(key, 
                       self.sub_name, str(self[key]), ''.join(self.sub_name)))
        except TypeError:
            self[key] = self.solver_namespace['prm'][key]
            info_green("Adding ['{0:s}'] = {1:s} to pdesubsystem {2:s}".format(key, 
                        str(self[key]), ''.join(self.sub_name)))
    else:
        try:
            self[key] = self.solver_namespace['prm'][key][self.sub_name]
            info_green("Adding ['{0:s}']['{1:s}'] = {2:s} to pdesubsystem {3:s}".format(key, 
                       self.sub_name, str(self[key]), ''.join(self.sub_name)))
        except:
            self[key] = self.solver_namespace['prm'][key]
            info_green("Adding ['{0:s}'] = {1:s} to pdesubsystem {2:s}".format(key, 
                       str(self[key]), ''.join(self.sub_name)))
    return self[key]

Subdict.__missing__ = __missing__

class NSSolver(PDESystem):
    
    def __init__(self, system_composition, problem, parameters):
        PDESystem.__init__(self, system_composition, problem, parameters)
        self.correction = None         # For Reynolds stress models
        self.resultfile = {}           # Files used to store the solution
        
    def setup(self):
        PDESystem.setup(self)
        self.problem.NS_solver = self # rename, same as problem.pdesystems['Navier-Stokes']
        # Create some short forms
        try:
            self.nuM = Constant(self.problem.prm['viscosity'])
            self.nu = self.nuM        
        except TypeError, KeyError:
            error('Viscosity not set in problem parameters')
            
        try:
            self.eps = Constant(self.prm['stabilization_prm'])
            self.convection_form = self.prm['convection_form']
        except TypeError, KeyError:
            error('Use solver parameters dictionary in top of this file to initialize class')
            
        self.n = FacetNormal(self.mesh)
        self.nut_ = None
        # Set coefficients for initial velocity and pressure
        if not self.problem.initialize(self):
            info_red('Initialization not performed for ' + self.prm['familyname'])
            
        # Get the body forces from the problem class
        try:
            self.f = self.problem.body_force()
        except NotImplementedError:
            pass

        # Get the list of boundaries from the problem class
        self.boundaries = self.problem.boundaries
                         
        # Generate boundary conditions using provided boundaries list
        self.bc = self.create_BCs(self.boundaries)
        
        # Rename testfunctions to enable use of the common 
        # testfunctions v and q for velocity and pressure. 
        try:
            self.v, self.q = self.vt['u'], self.vt['p']
        except:
            pass
        
        self.define()
                    
    def setup_subsystems(self):
        PDESystem.setup_subsystems(self)
        self.dim = self.problem.mesh.geometry().dim()
        # Symmetric tensorfunction space for stresses Sij
        self.prm['symmetry']['Sij'] = dict(((i,j), (j,i)) 
            for i in range(self.dim) for j in range(self.dim) if i > j )
        self.V['Sij'] = TensorFunctionSpace(self.mesh, 
            self.prm['family']['Sij'], self.prm['degree']['Sij'], 
            symmetry=self.prm['symmetry']['Sij'])
        self.S = self.V['Sij']
        self.qt['Sij'] = TrialFunction(self.S)
        self.vt['Sij'] = TestFunction(self.S)        

    def solve_derived_quantities(self):
        """Solve for parameters in list pdesubsystems['derived quantities']."""
        if 'derived quantities' in self.pdesubsystems:
            for pdesubsystem in self.pdesubsystems['derived quantities']:
                pdesubsystem.solve()

    def update(self):
        """Called at end of one iteration over pdesubsystems.
        This function is probably just intended for steady simulations
        where one wants to plot intermediate iterations to see how the
        simulation evolves"""
        if self.prm['plot_velocity']:
            if not self.problem.prm['time_integration'] == 'Steady':
                info_red('Set plot_velocity/pressure in problem_parameters to plot at end of timestep')
            if isinstance(self.u_, ufl.tensors.ListTensor):
                V = VectorFunctionSpace(self.mesh, 'CG', self.prm['degree']['u0'])
                u_ = project(self.u_, V)
                if hasattr(self, 'u_plot'):
                    self.u_plot.vector()[:] = u_.vector()[:]
                else:
                    self.u_plot = u_
            else:
                self.u_plot = self.u_
            plot(self.u_plot, rescale=True)
            
        if self.prm['plot_pressure']: plot(self.NS_solver.p_, title='Pressure',
                                           rescale=True)
