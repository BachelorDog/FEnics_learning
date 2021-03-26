__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2011-06-28"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU GPL version 3 or any later version"

from cbc.pdesys import *

class drivencavity_Solver(PDESystem):
    """Simple example of Navier-Stokes (NS) solvers using the PDESystem 
    class. Problem is a lid-driven cavity that can be either transient or 
    steady. To make it more interesting we also add a passive scalar 'c' to the 
    problem. The problem thus consist of two PDE subsystems: ['u', 'p'] for NS and 
    ['c'] for the passive scalar.
    """
    def __init__(self, mesh, parameter):
        PDESystem.__init__(self, [['u', 'p'], ['c']], mesh, parameters)
        self.nu = Constant(self.prm['viscosity'])
        self.dt = Constant(self.prm['dt'])
        self.f = Constant((0, 0))
        
        # Initialize the problem
        up0 = interpolate(Constant((0, 0, 0)), self.V['up'])
        c0 = Expression("n*n/(2.*pi)*exp(-0.5*n*n*(pow(x[0]-x0, 2) \
                                                 + pow(x[1]-y0, 2)))",
                        n=10, x0=0.5, y0=0.5)
        c0 = interpolate(c0, self.V['c'])
        self.initialize(dict(up=up0, c=c0))
        self.normalize['up'] = extended_normalize(self.V['up'], 2)
        # Define boundary conditions. Only natural bcs for c
        self.bc['up'] = [DirichletBC(self.V['up'].sub(0), (0., 0.), "on_boundary"),
                         DirichletBC(self.V['up'].sub(0), (1., 0.), "std::abs(x[1] - 1.) < DOLFIN_EPS && on_boundary")]
                         
        self.define()
        
    def define(self):
        ti = self.prm['time_integration']
        up_name = 'NS_{}_{}'.format(ti, self.prm['pdesubsystem']['up'])
        c_name  = 'Scalar_{}_{}'.format(ti, self.prm['pdesubsystem']['c'])
        
        self.add_pdesubsystem(eval(up_name), ['u', 'p'], bcs=self.bc['up'],
                              normalize=self.normalize['up'])
                              
        self.add_pdesubsystem(eval(c_name), ['c'])
        
    def update(self):
        plot(self.u_, rescale=True)
        plot(self.c_, rescale=True)
        info_red('        Total amount of c = {}'.format(assemble(self.c_*dx)))

class NS_Steady_1(PDESubSystem):
    """Implicit steady Navier-Stokes solver."""
    def form(self, u_, u, v_u, p, v_p, nu, f, **kwargs):
        return self.conv(v_u, u, u_)*dx + nu*inner(grad(v_u), grad(u))*dx \
            - inner(div(v_u), p)*dx - inner(v_p, div(u))*dx - inner(v_u, f)*dx
            
class NS_Steady_2(PDESubSystem):
    """Explicit steady Navier-Stokes solver"""
    def form(self, u_, u, v_u, p, v_p, nu, f, **kwargs):
        self.prm['reassemble_lhs'] = False
        return self.conv(v_u, u_, u_)*dx + nu*inner(grad(v_u), grad(u))*dx \
            - inner(div(v_u), p)*dx - inner(v_p, div(u))*dx - inner(v_u, f)*dx

class NS_Transient_1(PDESubSystem):
    """Transient form with Crank-Nicholson (CN) diffusion and where convection 
    is computed using AB-projection for convecting and CN for convected 
    velocity."""    
    def form(self, u, v_u, p, v_p, u_, nu, dt, u_1, u_2, f, **kwargs):
        U = 0.5*(u + u_1)
        U1 = 1.5*u_1 - 0.5*u_2
        F = (1./dt)*inner(u - u_1, v_u)*dx + self.conv(v_u, U, U1)*dx \
            + nu*inner(grad(v_u), grad(U) + grad(U).T)*dx \
            - inner(div(v_u), p)*dx - inner(v_p, div(u))*dx - inner(v_u, f)*dx
        return F
        
class Scalar_Transient_1(PDESubSystem):
    def form(self, c, v_c, c_, c_1, u, u_, u_1, u_2, nu, dt, **kwargs):
        U_ = 0.5*(u_ + u_1)
        C = 0.5*(c + c_1)
        F = (1./dt)*inner(c - c_1, v_c)*dx + inner(dot(U_, grad(C)), v_c)*dx \
            + nu*inner(grad(v_c), grad(C))*dx
        return F

class Scalar_Steady_1(PDESubSystem):
    """Pseudo-steady form for scalar"""
    def form(self, c, v_c, c_, u, u_, nu, dt, **kwargs):
        F = (1./dt)*inner(c - c_, v_c)*dx + inner(dot(u_, grad(c)), v_c)*dx \
            + nu*inner(grad(v_c), grad(c))*dx
        return F
        
if __name__=='__main__':
    # Define some parameters. default_parameters is found in PDESystem
    parameters = recursive_update(solver_parameters, {
        'viscosity': 0.01,
        'dt': 0.01,
        'T': 1.,
        'degree': {'u':2, 'c': 2},
        'space': {'u': VectorFunctionSpace},
        'time_integration': 'Transient'
    })
    mesh = UnitSquare(10, 10)
    solver = drivencavity_Solver(mesh, parameters)
    solver.solve(max_iter=1, redefine=False)
    interactive()
    #