__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-08-30"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"
"""
Turbulent channel flow.

For turbulent flows it is common to characterize the flow using Re_tau, 
a Reynolds number based on friction velocity u_tau=sqrt(nu*du/dy)_wall.
        
        Re_tau = u_tau/nu
        
The constant pressure gradient is computed by integrating the NS-equations
across the height of the channel

\int_{-1}^{1}(dp/dx) dV = -\int_{-1}^{1} nu*d^2/dy^2u dV 
2*dp/dx = - nu*(du/dy_top- du/dy_bottom) = - (-u_tau**2 - u_tau**2)
dp/dx = u_tau**2
"""

# import the channel mesh and parameters from the laminar problem case
from cbc.cfd.problems.channel import *
from cbc.cfd import icns                    # Navier-Stokes solvers
from cbc.cfd import ransmodels              # RANS models
from cbc.cfd.icns import solver_parameters  # parameters for NS
from cbc.cfd.ransmodels import solver_parameters as rans_parameters # parameters for RANS model
set_log_active(True)

from time import time
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from pylab import zeros, linspace
import cPickle

# Postprocessing
def tospline(problem): 
    """Store the results in a dictionary of splines and save to file.
    The channel solution is very often used to initialize other problems
    that have a VelocityInlet."""
    NS_solver = problem.pdesystems['Navier-Stokes']
    Turb_solver = problem.pdesystems['Turbulence model']
    f = open('../data/channel_' + problem.prm['turbulence_model'] + '_' + 
                str(problem.prm['Re_tau']) + '.ius','w')
    N = 1000
    xy = zeros(N)
    xy = zeros((N, 2))
    xy[:, 1] = linspace(0., 1., 1000) # y-direction
    spl = {}
    for s in (NS_solver, Turb_solver):            
        for name in s.names:
            vals = zeros((N, s.V[name].num_sub_spaces() or 1))
            for i in range(N):
                getattr(s, name + '_').eval(vals[i, :], xy[i, :])
            spl[name] = []
            # Create spline for all components
            # Scalar has zero subspaces, hence the or
            for i in range(s.V[name].num_sub_spaces() or 1): 
                spl[name].append(ius(xy[:, 1], vals[:, i]))
        
    cPickle.dump(spl, f)
    f.close()
    return spl

if __name__=='__main__':
    # Set up turbulent channel problem
    # Turbulent channel only works with periodic boundaries
    problem_parameters['periodic'] = True
    problem_parameters['time_integration'] = 'Steady'
    problem_parameters['Re_tau'] = Re_tau= 395.
    problem_parameters['utau'] = utau = 0.05
    problem_parameters['plot_velocity'] = True
    problem_parameters['Ny'] = 100
    problem_parameters['Nx'] = 10
    problem = channel(problem_parameters)
    problem.prm['viscosity'] = utau/Re_tau
    problem.pressure_gradient = Constant((utau**2, 0.)) # turbulent pressure gradient
    # Update the dictionary used for initialization.
    problem.q0.update(dict(
        k=0.01, 
        e=0.01, 
        v2=('0.001'),
        f=('1.'), 
        R=('0.001',)*3,
        Rij=('0.0',)*4, 
        Fij=('0.0',)*4))

    ## Set up Navier-Stokes solver ##
    solver_parameters['degree']['u'] = 1
    solver_parameters['omega'].default_factory = lambda : 0.8
    NS_solver = icns.NSCoupled(problem, solver_parameters)

    ## Set up turbulence model ##
    problem_parameters['turbulence_model'] = 'StandardKE'
    rans_parameters['omega'].default_factory = lambda : 0.7
    Turb_solver = ransmodels.StandardKE_Coupled(problem, rans_parameters,
                            model=problem_parameters['turbulence_model'])

    ## solve the problem ##    
    t0 = time()
    problem_parameters['max_iter'] = 100
    problem.solve()
    print('time = ', time()-t0)
    print(list_timings())
    plot(NS_solver.u_)

