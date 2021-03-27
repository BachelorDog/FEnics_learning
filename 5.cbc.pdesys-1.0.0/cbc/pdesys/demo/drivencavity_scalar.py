"""This demo illustrates how to use of DOLFIN and CBC.PDESys
for solving the Navier-Stokes equations together with two passive
scalars """

__author__ = "Mikael Mortensen <mikael.mortensen@gmail.com>"
__date__ = "2011-11-18"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU GPL version 3 or any later version"

import sys
from cbc.pdesys import *

class NavierStokes(PDESubSystem):
    """Transient form with Crank-Nicolson (CN) diffusion and where convection 
    is computed using AB-projection for convecting and CN for convected 
    velocity."""    
    def form(self, u, v_u, p, v_p, u_, nu, dt, u_1, u_2, f, **kwargs):
        U = 0.5*(u + u_1)
        U1 = 1.5*u_1 - 0.5*u_2
        F = (1./dt)*inner(u - u_1, v_u)*dx + self.conv(v_u, U, U1)*dx \
            + nu*inner(grad(v_u), grad(U) + grad(U).T)*dx \
            - inner(div(v_u), p)*dx - inner(v_p, div(u))*dx - inner(v_u, f)*dx
        return F
        
class Scalar(PDESubSystem):
    """Passive scalar with Crank-Nicolson diffusion and convection
    """
    def form(self, c, v_c, c_, c_1, u_, u_1, nu, Pr, dt, **kwargs):
        U_ = 0.5*(u_ + u_1)
        C = 0.5*(c + c_1)
        F = (1./dt)*inner(c - c_1, v_c)*dx + inner(dot(U_, grad(C)), v_c)*dx \
            + nu/Pr*inner(grad(v_c), grad(C))*dx
        return F

class NavierStokesScalar(PDESubSystem):
    """Fully coupled transient Navier-Stokes solver plus one passive scalar
    """    
    def form(self, u, v_u, p, v_p, u_, nu, dt, u_1, u_2, f,
             c, v_c, c_, c_1, c_2, Pr, **kwargs):
        
        if not self.prm['iteration_type'] == 'Newton':
            info_red('Scheme is not linearized and requires Newton iteration type')
            
        U = 0.5*(u + u_1)
        C = 0.5*(c + c_1)
        U1 = 1.5*u_1 - 0.5*u_2
        C1 = 1.5*c_1 - 0.5*c_2
        
        Fu = (1./dt)*inner(u - u_1, v_u)*dx + self.conv(v_u, U, U1)*dx \
            + nu*inner(grad(v_u), grad(U) + grad(U).T)*dx \
            - inner(div(v_u), p)*dx - inner(v_p, div(u))*dx - inner(v_u, f)*dx
            
        Fc = (1./dt)*inner(c - c_1, v_c)*dx + inner(dot(U1, grad(C)), v_c)*dx \
            + nu/Pr*inner(grad(v_c), grad(C))*dx
        
        return Fu + Fc

# Set up driven cavity problem
problem_parameters = recursive_update(problem_parameters, {
    'viscosity': 0.01,
    'dt': 0.005,
    'T': 0.5,
    'time_integration': 'Transient'
})
mesh = UnitSquare(20, 20)
problem = Problem(mesh, problem_parameters)

fully_coupled = eval(sys.argv[-1])

# Set up Navier-Stokes PDE system
solver_parameters = recursive_update(solver_parameters, {
    'degree': {'u':2, 'c': 2},
    'space': {'u': VectorFunctionSpace},
})

if fully_coupled:
    
    solver_parameters['familyname'] = 'Navier-Stokes-Scalar'    
    NS = PDESystem([['u', 'p', 'c']], problem, solver_parameters)
    NS.nu = Constant(problem.prm['viscosity'])
    NS.f = Constant((0, 0))
    NS.Pr = Constant(1.)
    normalization = extended_normalize(NS.V['upc'], 2)
    bcs = [DirichletBC(NS.V['u'], (0., 0.), "on_boundary"),
           DirichletBC(NS.V['u'], (1., 0.), "on_boundary && x[1] > 1. - DOLFIN_EPS"),
           DirichletBC(NS.V['upc'].sub(2), (1.), "on_boundary && x[1] > 1. - DOLFIN_EPS")]

    for bc in bcs:
        bc.apply(NS.upc_.vector())
        bc.apply(NS.upc_1.vector())
    
    NS.add_pdesubsystem(NavierStokesScalar, ['u', 'p', 'c'], 
                         bcs=bcs, normalize=normalization)
                                        
    def update(self):
        sol = self.pdesystems['Navier-Stokes-Scalar']
        plot(sol.u_, rescale=True)
        plot(sol.c_, rescale=True)
        info_red('        Scalar1: Total amount of c = {}'.format(assemble(sol.c_*dx)))
    Problem.update = update
    
    problem.prm['T'] = 2.0
    problem.solve()

else:
    solver_parameters['familyname'] = 'Navier-Stokes'
    NS = PDESystem([['u', 'p']], problem, solver_parameters)
    NS.nu = Constant(problem.prm['viscosity'])
    NS.f = Constant((0, 0))
    normalization = extended_normalize(NS.V['up'], 2)
    bcs = [DirichletBC(NS.V['u'], (0., 0.), "on_boundary"),
        DirichletBC(NS.V['u'], (1., 0.), "on_boundary && x[1] > 1. - DOLFIN_EPS")]
    
    NS.add_pdesubsystem(NavierStokes, ['u', 'p'], bcs=bcs, normalize=normalization)

    # Overload update method to plot intermediate results. update is called at the end of each timestep
    def update(self):
        plot(self.pdesystems['Navier-Stokes'].u_, rescale=True)
        if len(self.pdesystemlist) > 1: # If scalars are defined yet
            for i, name in enumerate(self.pdesystemlist[1:]):
                sol = self.pdesystems[name]
                plot(sol.c_, rescale=True)
                info_red('        {}: Total amount of c = {}'.format(name, assemble(sol.c_*dx)))
    Problem.update = update

    # Integrate solution a few timesteps up to prm['T']
    #problem.solve()

    # Set up two scalar PDESystems with different diffusivity and boundary conditions
    solver_parameters['familyname'] = 'Scalar1'
    Scalar1 = PDESystem([['c']], problem, solver_parameters)
    solver_parameters['familyname'] = 'Scalar2'
    Scalar2 = PDESystem([['c']], problem, solver_parameters)

    # Scalar 1 is unity on the top boundary, whereas scalar 2 is unity on the bottom
    bcs1 = [DirichletBC(Scalar1.V['c'], (1.), "on_boundary && x[1] > 1. - DOLFIN_EPS")]
    bcs2 = [DirichletBC(Scalar2.V['c'], (1.), "on_boundary && x[1] < DOLFIN_EPS")]
    Scalar1.nu = Scalar2.nu = Constant(problem.prm['viscosity'])
    Scalar1.Pr = Constant(1.)
    Scalar2.Pr = Constant(2.)
    Scalar1.u_  = Scalar2.u_  = NS.u_
    Scalar1.u_1 = Scalar2.u_1 = NS.u_1

    # Hook up the scalar transport form
    Scalar1.add_pdesubsystem(Scalar, ['c'], bcs=bcs1)
    Scalar2.add_pdesubsystem(Scalar, ['c'], bcs=bcs2)

    # Integrate the solution up to prm['T']
    problem.prm['T'] = 2.
    problem.solve()
