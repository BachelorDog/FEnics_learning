__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-10-17"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"
"""
Orr Sommerfeld equation
This problem simulates the growth of an instability in plane channel flow.
The eigenvalue and eigenvector of the instability is computed by solving
the Orr Sommerfeld equation in a channel ranging -1 < y < 1,
using the analytical solution for laminar flow U=1-y**2.
The Orr Sommerfeld equation is derived from the Navier Stokes equations
assuming a periodic disturbance that is proportional to exp(iA(x-ct)),
where A is a parameter and lambda = -iAc are the eigenvalues.

Having computed the Orr Sommerfeld equation we are looking at the only unstable
eigenvalue and its corresponding eigenvector. From these we can create the
streamfunction for the perturbation psi = phi(y)*exp(iA(x-ct)), which in turn is
used to create an initial velocity field

U = 1 - y^2 + eps*real{phi'(y)*exp(iA(x-ct))}
V = -eps*real{i*A*phi(y)*exp(iA(x-ct))}

Here eps is the amplitude of the disturbance and it must be small.
This small perturbation grows in time and the solution is known analytically.
"""
from channel import *
from cbc.cfd.tools.OrrSommerfeld_eig import OrrSommerfeld
from cbc.cfd import icns                    # Navier-Stokes solvers
from cbc.cfd.icns import solver_parameters  # parameters to NS solver
set_log_active(True)

from numpy import real, exp, dot as ndot

class U0(Expression):
    """ Expression for the Orr-Sommerfeld solution """
    def __init__(self, OS, **kwargs):
        self.par={  
            'eps':1.e-4,
            't':0.,
            'theta':1.
        }
        self.par.update(kwargs)
        self.OS = OS
        [setattr(self, name, val) for name, val in self.par.iteritems()]
        
    def eval(self, values, x):
        self.OS.interp(x[1])
        values[0] = self.theta*(1.-x[1]**2) + self.eps*ndot(self.OS.f, 
                         real(self.OS.dphidy*exp(1j*(x[0] - self.OS.eigval*self.t))))
        values[1] = -self.eps*ndot(self.OS.f, real(1j*self.OS.phi*exp(1j*(x[0] - 
                         self.OS.eigval*self.t))))

    def value_shape(self):
        return (2,)
        
class UP0(Expression):
    """ Expression for the Orr-Sommerfeld solution """
    def __init__(self, OS, **kwargs):
        self.par={  
            'eps':1.e-4,
            't':0.,
            'theta':1.
        }
        self.par.update(kwargs)
        self.OS = OS
        [setattr(self, name, val) for name, val in self.par.iteritems()]
        
    def eval(self, values, x):
        self.OS.interp(x[1])
        values[0] = self.theta*(1.-x[1]**2) + self.eps*ndot(self.OS.f, 
                         real(self.OS.dphidy*exp(1j*(x[0] - self.OS.eigval*self.t))))
        values[1] = -self.eps*ndot(self.OS.f, real(1j*self.OS.phi*exp(1j*(x[0] - 
                         self.OS.eigval*self.t))))
        values[2] = 0

    def value_shape(self):
        return (3,)

class Ue(Expression):
    """ Exact laminar velocity """
    def eval(self, values, x):
        values[0] = (1.-x[1]**2)
        values[1] = 0.
        
    def value_shape(self):
        return (2,)

# Default parameters for channel
problem_parameters['periodic'] = True
problem_parameters['Nx'] = 20
problem_parameters['Ny'] = 50
problem_parameters['Re'] = 8000.
problem_parameters['L'] = 2.*pi
problem_parameters['time_integration'] = 'Transient'
problem_parameters['T'] = 0.5
problem_parameters['max_iter'] = 1          # iterations per timestep
problem_parameters['plot_velocity'] = False # plot velocity at end of timestep
problem_parameters['periodic'] = False      # Use or not periodic boundary conditions

solver_parameters = recursive_update(solver_parameters, 
dict(degree=dict(u=1),
    pdesubsystem=dict(u=1, p=1, velocity_update=1, up=1),
    linear_solver=dict(u='bicgstab', p='gmres', velocity_update='bicgstab'), 
    precond=dict(u='jacobi', p='amg', velocity_update='jacobi'),
    convection_form='Divergence')
)

NSchannel = channel
class channel(NSchannel):
    
    def initialize(self, pdesystem):
        """Initialize with the Orr-Sommerfeld solution."""
        self.q0 = OS_velocity
        self.ue = interpolate(Ue(), pdesystem.V['u'])
        up = interpolate(self.q0['u'], pdesystem.V['u'])
        self.e0 = 0.5*norm(up)**2
        self.e1 = 0.5*errornorm(up, self.ue)**2
        return NSchannel.initialize(self, pdesystem)

    def update(self):
        info_green('Relative kinetic energy = ' + 
          str(0.5*errornorm(self.pdesystems['Navier-Stokes'].u_, self.ue)**2/self.e1))

# Solve the Orr-Sommerfeld eigenvalue problem
OS = OrrSommerfeld(Re=problem_parameters['Re'], N=60)

# Create a dictionary for initializing NS solvers
OS_velocity = Initdict(u = U0(OS=OS),
                       up = UP0(OS=OS),
                       p = 0.0)

# Set up problem
problem = channel(problem_parameters)
problem_parameters['dt'] = 0.01
problem.q0 = OS_velocity

# Choose Navier-Stokes solver
#solver = icns.NSSegregated(problem, solver_parameters)
solver = icns.NSCoupled(problem, solver_parameters)

# Solve the problem
problem.solve()
plot(solver.u_)

# Check where the time went
print summary()

