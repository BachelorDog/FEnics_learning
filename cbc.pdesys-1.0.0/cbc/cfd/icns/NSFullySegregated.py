__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-10-01"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"

from NSSolver import *

class NSFullySegregated(NSSolver):
    """Segregated solver for the Navier-Stokes equations.
    The velocity vector is implemented as individual components u0, u1 (u2)
    using FunctionSpaces and not a VectorFunctionSpace. 
    The velocity-components all make use of the same coefficient matrix.
    """
   
    def __init__(self, problem, parameters):
        self.dim = problem.mesh.geometry().dim()
        if self.dim == 2:
            sys_comp = [['u0'], ['u1'], ['p']]
            self.u_components = ['u0', 'u1']
        else:
            sys_comp = [['u0'], ['u1'], ['u2'], ['p']]
            self.u_components = ['u0', 'u1', 'u2']
        NSSolver.__init__(self, system_composition=sys_comp,
                                problem=problem, 
                                parameters=parameters)
   
    def define(self):
        # Rename and set up solution vectors
        self.u = self.qt['u0']
        self.v = self.vt['u0']
        self.q = self.vt['p']
        
        # Create vector views
        self.u_ = as_vector([self.q_[ui] for ui in self.u_components])
        if self.prm['time_integration'] == 'Transient':
            self.u_1 = as_vector([self.q_1[ui] for ui in self.u_components])
            self.u_2 = as_vector([self.q_2[ui] for ui in self.u_components])
            
        # Hook up the pdesubsystems for velocity, pressure and velocity update
        # Use the 'u' and 'velocity_update' keys for pdesubsystem for simplicity
        u_subsystem = self.prm['time_integration'] + '_Velocity_' + \
                      str(self.prm['pdesubsystem']['u'])
            
        for ui in self.u_components: 
            self.add_pdesubsystem(eval(u_subsystem), [ui], bcs=self.bc[ui])
                
        p_subsystem = self.prm['time_integration'] + '_Pressure_' + \
                      str(self.prm['pdesubsystem']['p'])
        self.normalize['p'] = extended_normalize(self.V['p'], part='whole')
        #self.normalize['p'] = dolfin_normalize
        
        self.add_pdesubsystem(eval(p_subsystem), ['p'], bcs=self.bc['p'],
                              normalize=self.normalize['p'])
        
        # Initialize pressure through this presolve (should improve accuracy of first step)
        #self.pdesubsystems['p'].solve()
        if self.prm['time_integration'] == 'Transient':
            uu_subsystem = 'VelocityUpdate_' + \
                           str(self.prm['pdesubsystem']['velocity_update'])
            for ui in self.u_components:
                self.pdesubsystems[ui + '_update'] = eval(uu_subsystem)(vars(self), [ui],
                            bcs=self.bc[ui],
                            precond=self.prm['precond']['velocity_update'],
                            linear_solver=self.prm['linear_solver']['velocity_update'])
  
    def create_BCs(self, bcs):
        """
        Create boundary conditions for velocity and pressure based on 
        boundaries in list bcs.
        """
        bcu = dict((ui, []) for ui in self.u_components + ['p'])
        for bc in bcs:
            if bc.type() in ('VelocityInlet', 'Wall'):
                if hasattr(bc, 'func'): # Use func if provided
                    for ui in self.u_components:
                        add_BC(bcu[ui], self.V[ui], bc, bc.func[ui])
                else:
                    # Default is zero on walls for all quantities
                    if bc.type() == 'Wall':
                        for ui in self.u_components:
                            add_BC(bcu[ui], self.V[ui], bc, Constant(0))
                    elif bc.type() == 'VelocityInlet':
                        raise TypeError('expected func for VelocityInlet')
                if bc.type() == 'VelocityInlet':
                    bcu['p'].append(bc)
            elif bc.type() in ('ConstantPressure'):
                """ This bc is weakly enforced for u """
                for ui in self.u_components:
                    bcu[ui].append(bc)
                add_BC(bcu['p'], self.V['p'], bc, bc.func['p'])
            elif bc.type() in ('Outlet', 'Symmetry'):
                for ui in self.u_components:
                    bcu[ui].append(bc)
                add_BC(bcu['p'], self.V['p'], bc, bc.func['p'])
            elif bc.type() == 'Periodic':
                for ui in self.u_components:
                    add_BC(bcu[ui], self.V[ui], bc, None)
                add_BC(bcu['p'], self.V['p'], bc, None)
            else:
                info("No assigned boundary condition for %s -- skipping..." \
                     %(bc.__class__.__name__))
        return bcu

    def Transient_update(self):
        # Update to new time-level
        for ui in self.system_names[:-1]:
            dummy = self.pdesubsystems[ui + '_update'].solve()
        NSSolver.Transient_update(self)
        
######### Factory functions or classes for numerical pdesubsystems ###################
 
class VelocityBase(PDESubSystem):
    """Variational form for velocity."""    
    def get_solver(self):
        #Use the same solver for all components
        self.index = eval(self.name[-1])     # Velocity component
        if self.index > 0:
            return self.solver_namespace['pdesubsystems']['u0'].linear_solver
        else:
            return PDESubSystem.get_solver(self)
    
    def assemble(self, M):
        # For u1, u2 use the assembled matrix of u0
        if isinstance(M, Matrix) and self.index > 0:
            self.A = self.solver_namespace['pdesubsystems']['u0'].A
        else:
            PDESubSystem.assemble(self, M)
                    
    def define(self):
        form_args = self.solver_namespace.copy()
        self.get_form(form_args)
        if self.F:
            self.a, self.L = lhs(self.F), rhs(self.F)
                   
class VelocityUpdateBase(PDESubSystem):
    """Variational form for velocity update using constant mass matrix."""
    def __init__(self, solver_namespace, unknown, bcs=[], **kwargs):
        PDESubSystem.__init__(self, solver_namespace, unknown, 
                                    bcs=bcs, **kwargs)
        self.prm['reassemble_lhs'] = False # Constant coefficient matrix
        
    def get_solver(self):
        # Use the same solver for all components
        self.index = eval(self.name[-1])     # Velocity component
        if self.index > 0:
            return self.solver_namespace['pdesubsystems']['u0_update'].linear_solver
        else:
            return PDESubSystem.get_solver(self)

    def assemble(self, M):
        if isinstance(M, Matrix) and self.index > 0:
            # Use the assembled matrix of u0 for u1 and u2
            self.A = self.solver_namespace['pdesubsystems']['u0_update'].A
        else:
            PDESubSystem.assemble(self, M)
            
    def define(self):
        self.get_form(self.solver_namespace)                
        if self.F:
            self.a, self.L = lhs(self.F), rhs(self.F)        
        
class PressureBase(PDESubSystem):
    """Pressure base class."""
    def __init__(self, solver_namespace, unknown, 
                       bcs=[], normalize=None, **kwargs):
        PDESubSystem.__init__(self, solver_namespace, unknown, bcs=bcs, 
                                    normalize=normalize, **kwargs)
        
        # Create functions for the pressure correction
        self.solver_namespace['dp'] = Function(self.V)
        self.solver_namespace['dpx'] = self.dpx = self.solver_namespace['dp'].vector()        
        self.prm['iteration_type'] = 'Picard'
        self.p_old = Function(self.V)
        self.px_old = self.p_old.vector()
        self.prm['reassemble_lhs'] = False # Constant coefficient matrix
        
    def define(self):
        if any([bc.type() in ['ConstantPressure', 'Outlet', 'Symmetry'] for bc in self.bcs]): 
            self.normalize = None # Because there is a Dirichlet on pressure
        
        self.get_form(self.solver_namespace)
        
        exterior = self.add_exterior(**self.solver_namespace)
        if exterior: # Check for ds term and add to F
            self.F = self.F + exterior
        
        self.a, self.L = lhs(self.F), rhs(self.F)

    def add_exterior(self, p, p_, q, n, dt, u_, nu, **kwargs):        
        L = []
        for bc in self.bcs:
            if bc.type() == 'Outlet':
                L.append(-inner(q*n, grad(p))*ds(bc.bid))
                self.exterior_facet_domains = bc.mf 
                
        if len(L) > 0:
            return reduce(operator.add, L)
        else:
            return False
                            
    def prepare(self):
        """Remember old pressure solution"""
        self.px_old[:] = self.x[:]
        
    def update(self):
        """Get pressure correction"""
        self.dpx[:] = self.x[:] - self.px_old[:]
        
############# Velocity update #################################
class VelocityUpdate_1(VelocityUpdateBase):
    """ Velocity update using constant mass matrix """
    def form(self, u_, u, v, dt, dp, p_, **kwargs):
        return inner(u - u_[self.index], v)*dx + inner(dt*dp.dx(self.index), v)*dx
        
class VelocityUpdate_101(VelocityUpdateBase):
    """ 
    Optimized version of VelocityUpdate_1.
    """    
    def form(self, v, p, p_, dt, dpx, **kwargs): 
        self.dt = dt(0)
        self.dpx = dpx
        # Assemble matrix used to compute rhs
        if self.x.size()==p_.vector().size(): # Reuse matrix to save memory
            self.P = self.solver_namespace['pdesubsystems']['u'+str(self.index)].P
        else:
            self.P = assemble(v * p.dx(self.index) * dx)            
        self.b = Vector(self.x)
        return False
        
    def assemble(self, M):
        # Use the assembled matrix of u0 or reuse M from momentum equation
        if self.index == 0:
            if any([bc.type() in ['Periodic'] for bc in self.bcs]):
                self.A = self.solver_namespace['pdesubsystems']['u0'].M.copy()
            else:
                self.A = self.solver_namespace['pdesubsystems']['u0'].M
            self.A.initialized = True
        else:
            self.A = self.solver_namespace['pdesubsystems']['u0_update'].A
        # This matrix could be compressed for additional speed-up
            
    def solve_Picard_system(self, assemble_A, assemble_b):
        if assemble_A: # Assemble only on first timestep
            self.assemble(self.A)
            [bc.apply(self.A) for bc in self.bcs]
            #self.A.compress()
        # Compute rhs using matrix-vector products
        self.b[:] = self.A*self.x        
        self.b.axpy(-self.dt, self.P*self.dpx)
        [bc.apply(self.b) for bc in self.bcs]
        
        # Update velocity
        self.setup_solver(assemble_A, assemble_b)
        self.linear_solver.solve(self.A, self.x, self.b)
        return 0., self.x

############# Velocity update #################################

############# Pressure ########################################
class Transient_Pressure_1(PressureBase):
    
    def form(self, p_, p, q, u_, dt, **kwargs):   
        return inner(grad(q), grad(p))*dx - inner(grad(q), grad(p_))*dx + \
               (1./dt)*q*div(u_)*dx
        
class Transient_Pressure_101(PressureBase):
    """ Optimized version of Transient_Pressure_1."""     
    def form(self, u, p, q, dt, dim, **kwargs): 
        # Preassemble matrices used to compute rhs
        self.R = [assemble(inner(q, u.dx(i))*dx) for i in range(dim)]
        self.dim = dim
        self.b = Vector(self.x)
        return inner(grad(q), dt*grad(p))*dx
        
    def solve_Picard_system(self, assemble_A, assemble_b):
        self.prepare()
        if assemble_A:
            self.assemble(self.A)
            [bc.apply(self.A) for bc in self.bcs]
            #self.A.compress()
        self.b[:] = self.A*self.x
        for i in range(self.dim):
            self.b.axpy(-1., self.R[i]*self.solver_namespace['u_'][i].vector()) # Divergence of u_

        [bc.apply(self.b) for bc in self.bcs]
        self.rp = residual(self.A, self.x, self.b)
        self.work[:] = self.x[:]
        self.setup_solver(assemble_A, assemble_b)
        self.linear_solver.solve(self.A, self.x, self.b)
        #if self.normalize: self.normalize(self.x)
        self.update()
        return self.rp, self.x - self.work

############# Pressure ########################################

############# Velocity ########################################
class Transient_Velocity_1(VelocityBase):
    """Incremental pressure correction.
    Crank-Nicholson (CN) diffusion. Convection is computed using 
    AB-projection for convecting and CN for convected velocity.
    """
    def form(self, u_, u, v, p_, u_1, u_2, nu, f, dt, convection_form, **kwargs):         
        U  = 0.5*(u + u_1[self.index])        
        U_ = 1.5*u_1 - 0.5*u_2        
        return (1./dt)*inner(u - u_1[self.index], v)*dx + \
               self.conv(v, U, U_, convection_form)*dx + \
               nu*inner(grad(U), grad(v))*dx + \
               inner(v, p_.dx(self.index))*dx \
               - inner(v, f[self.index])*dx

class Transient_Velocity_2(VelocityBase):
    """Incremental pressure correction.
    Crank-Nicholson (CN) diffusion. Convection is computed with explicit 
    Adams-Bashforth projection and the coefficient matrix can be preassembled.
    Scheme is linear and Newton returns the same as Picard.
    """
    def form(self, u_, u, v, p_, u_1, u_2, nu, f, dt, convection_form, **kwargs):    
        self.prm['reassemble_lhs'] = False
        U  = 0.5*(u + u_1[self.index])
        return (1./dt)*inner(u - u_1[self.index], v)*dx + \
               1.5*self.conv(v, u_1[self.index], u_1, convection_form)*dx - \
               0.5*self.conv(v, u_2[self.index], u_2, convection_form)*dx + \
               nu*inner(grad(U), grad(v))*dx + \
               inner(v, p_.dx(self.index))*dx - inner(v, f[self.index])*dx
            
class Transient_Velocity_101(VelocityBase):
    """ 
    Optimized version of Transient_Velocity_1
    """     
    def form(self, u_, u, v, p, q, p_, u_1, u_2, nu, f, dt, convection_form, dim, **kwargs): 
             
        if self.index == 0:
            # Assemble matrices that don't change
            self.M = assemble(inner(v, u)*dx)                # mass
            self.K = assemble(nu*inner(grad(v), grad(u))*dx) # diffusion
            U_ = 1.5*u_1 - 0.5*u_2                           # AB-projection
            self.a = 0.5*self.conv(v, u, U_, convection_form)*dx # convection form that changes in time
        self.dt = dt(0)
        self.x_1 = self.solver_namespace['x_1']           
        self.b0 = assemble(inner(f[self.index], v)*dx)    
        self.pdes = self.solver_namespace['pdesubsystems']
        self.P = assemble(inner(v, p.dx(self.index))*dx) # Different for each component
        #self.P = assemble(-inner(v.dx(self.index), p)*dx) # Different for each component
        # Set the initial rhs-vector equal to the constant body force.
        self.b = Vector(self.x)
        self.bold = Vector(self.x)
        
        # Do not reassemble lhs when iterating over pressure-velocity system on given timestep
        # This is possible because self.a does not depend on u_ (the new solution)
        self.prm['reassemble_lhs_inner'] = False
        self.exterior = False
        return False
            
    def assemble(self, *args):
        if self.index == 0: # Assemble only for first velocity component
            # Set up coefficient matrix                
            self.A = assemble(self.a, tensor=self.A, reset_sparsity=self.prm['reset_sparsity']) 
            self.A._scale(-1.) # Negative convection on the rhs 
            self.A.axpy(1./self.dt, self.M, True)    # Add mass
            self.A.axpy(-0.5, self.K, True)          # Add diffusion                
            
            # Compute rhs for all velocity components
            for ui in self.solver_namespace['u_components']:
                self.pdes[ui].b[:] = self.pdes[ui].b0[:]
                self.pdes[ui].b.axpy(1., self.A*self.x_1[ui])

            # Reset matrix for lhs
            self.A._scale(-1.)
            self.A.axpy(2./self.dt, self.M, True)
            self.prm['reset_sparsity'] = False 
            self.A.initialized = True
            ## FixMe. For some reason reset_sparsity must be True for periodic bcs ?? Perhaps the modified sparsity pattern in (1 -1) rows??
            if any([bc.type() == 'Periodic' for bc in self.bcs]):
                self.prm['reset_sparsity'] = True 
        else:
            self.A = self.pdes['u0'].A

    def solve_Picard_system(self, assemble_A, assemble_b):
        # Assemble A and parts of b
        if assemble_A: self.assemble()
        
        # In case of inner iterations over u-p system, it is only 
        # the pressure part of b that needs reassembling. Remember the
        # preassembled part in bold
        self.bold[:] = self.b[:] 
        self.b.axpy(-1., self.P*self.solver_namespace['x_']['p'])
        [bc.apply(self.A, self.b) for bc in self.bcs]
        self.work[:] = self.x[:]    # start vector for iterative solvers
        # Check out line below to save one matrix vector product
        rv = residual(self.A, self.x, self.b)
        #rv = 0
        self.setup_solver(assemble_A, assemble_b)
        self.linear_solver.solve(self.A, self.x, self.b)
        self.b[:] = self.bold[:]  # preassemble part
        return rv, self.x - self.work
        
class Transient_Velocity_102(VelocityBase):
    """ 
    Optimized version of Transient_Velocity_2
    """     
    def form(self, u_, u, v, p, q, p_, u_1, u_2, nu, f, dt, convection_form, dim, **kwargs):
        self.dt = dt(0)
        self.dim = dim
        self.ac1 = self.conv(v, u, u_1, convection_form)*dx
        self.ac2 = self.conv(v, u, u_2, convection_form)*dx
        self.Ac1 = Matrix()
        self.Ac2 = Matrix()
        self.x_1 = self.solver_namespace['x_1']
        self.x_2 = self.solver_namespace['x_2']
        self.pdes = self.solver_namespace['pdesubsystems']
        if self.index == 0:
            # Assemble matrices that don't change
            self.A = assemble(0.5*nu*inner(grad(v), grad(u))*dx)
            self.A.initialized = True
            self.M = assemble(inner(v, u)*dx)
            self.A.axpy(1./self.dt, self.M, True)    # Add mass
            [bc.apply(self.A) for bc in self.bcs]
        else:
            self.A = self.pdes['u0'].A
            
        self.P = assemble(inner(v, p.dx(self.index))*dx)
        # Set the initial rhs-vector equal to the constant body force.
        self.b = Vector(self.x)
        self.bold = Vector(self.x)
        self.b0 = assemble(inner(f[self.index], v)*dx)        
        self.exterior = False
        self.prm['reassemble_rhs_inner'] = False # Call assemble(b) only in first inner iter
        return False
            
    def assemble(self, *args):
        if self.index == 0: # Assemble only for first velocity component
            # Set up coefficient matrix                
            self.A._scale(-1.) # Negative diffusion on the rhs 
            self.A.axpy(2./self.dt, self.M, True)  # -1/dt*M + 2/dt*M = 1/dt*M
            
            # Compute rhs for all velocity components
            self.Ac1 = assemble(self.ac1, tensor=self.Ac1, 
                                reset_sparsity=self.prm['reset_sparsity'])
            self.Ac2 = assemble(self.ac2, tensor=self.Ac2, 
                                reset_sparsity=self.prm['reset_sparsity'])
            for ui in self.solver_namespace['u_components']:
                self.pdes[ui].b[:] = self.pdes[ui].b0[:]
                self.pdes[ui].b.axpy(1., self.A*self.x_1[ui])
                self.pdes[ui].b.axpy(-1.5, self.Ac1*self.x_1[ui])
                self.pdes[ui].b.axpy( 0.5, self.Ac2*self.x_2[ui])
            # Reset matrix for lhs
            self.A._scale(-1.)
            self.A.axpy(2./self.dt, self.M, True)
            self.prm['reset_sparsity'] = False

    def solve_Picard_system(self, assemble_A, assemble_b):
        """One assemble and solve of Picard system."""
        # For explicit convection it is only the rhs that needs reassembling
        if assemble_b: self.assemble(self.b)
        # In case of inner iterations over u-p system, it is only 
        # the pressure part of b that needs reassembling. Remember the
        # preassembled part in bold
        self.bold[:] = self.b[:] 
        self.b.axpy(-1., self.P*self.solver_namespace['x_']['p'])
        [bc.apply(self.b) for bc in self.bcs]
        self.work[:] = self.x[:]    # start vector for iterative solvers
        # Check out line below to save one matrix vector product
        rv = residual(self.A, self.x, self.b)
        #rv = 0
        self.setup_solver(assemble_A, assemble_b)
        self.linear_solver.solve(self.A, self.x, self.b)
        self.b[:] = self.bold[:]  # preassembled part
        return rv, self.x - self.work
        