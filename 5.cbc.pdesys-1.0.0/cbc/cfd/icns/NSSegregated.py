__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-08-30"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"

from NSSolver import *

class NSSegregated(NSSolver):
   
    def __init__(self, problem, parameters):
        NSSolver.__init__(self, system_composition=[['u'], ['p']],
                                problem=problem, 
                                parameters=parameters)   
                            
    def define(self):
        classname = self.prm['time_integration'] + '_Velocity_' + \
                    str(self.prm['pdesubsystem']['u'])
        self.pdesubsystems['u'] = eval(classname)(vars(self), ['u'], bcs=self.bc['u'])
                
        classname = self.prm['time_integration'] + '_Pressure_' + \
                    str(self.prm['pdesubsystem']['p'])
        self.normalize['p'] = extended_normalize(self.V['p'], part='whole')
        self.pdesubsystems['p'] = eval(classname)(vars(self), ['p'], bcs=self.bc['p'],
                                                  normalize=self.normalize['p'])
        
        if (self.prm['time_integration'] == 'Transient' and not
            self.prm['pdesubsystem']['velocity_update'] == 0):
            classname = 'VelocityUpdate_' + \
                str(self.prm['pdesubsystem']['velocity_update'])
            self.pdesubsystems['velocity_update'] = eval(classname)(vars(self), ['u'], 
                            bcs=self.bc['u'],
                            precond=self.prm['precond']['velocity_update'],
                            linear_solver=self.prm['linear_solver']['velocity_update'])
                    
        self.pdesubsystems['p'].solve()
  
    def create_BCs(self, bcs):
        """
        Create boundary conditions for velocity and pressure
        based on boundaries in list bcs.
        """
        bcu = {'u': [], 'p': [], 'component': []}
        V1 = FunctionSpace(self.mesh, self.prm['family']['u'], self.prm['degree']['u'])
        for bc in bcs:
            if bc.type() in ('VelocityInlet', 'Wall'):
                if hasattr(bc, 'func'): # Use func if provided
                    add_BC(bcu['u'], self.V['u'], bc, bc.func['u'])
                    # Only used for A, so value is not interesting
                    add_BC(bcu['component'], V1, bc, Constant(0))  
                else:
                    if bc.type() == 'Wall':
                        # Default is zero on walls for all quantities
                        func = Constant((1e-12, )*self.mesh.geometry().dim())
                        add_BC(bcu['u'], self.V['u'], bc, func)
                        add_BC(bcu['component'], V1, bc, Constant(0))
                    elif bc.type() == 'VelocityInlet':
                        raise TypeError('expected func for VelocityInlet')
                if bc.type() == 'VelocityInlet':
                    bcu['p'].append(bc)
            elif bc.type() in ('ConstantPressure'):
                """ This bc is weakly enforced for u """
                bcu['u'].append(bc)
                bcu['component'].append(bc)
                add_BC(bcu['p'], self.V['p'], bc, bc.func['p'])
            elif bc.type() in ('Outlet', 'Symmetry'):
                bcu['u'].append(bc)
                add_BC(bcu['p'], self.V['p'], bc, bc.func['p'])
            elif bc.type() == 'Periodic':
                add_BC(bcu['u'], self.V['u'], bc, None)
                add_BC(bcu['p'], self.V['p'], bc, None)
                add_BC(bcu['component'], V1, bc, None)                
            else:
                info("No assigned boundary condition for %s -- skipping..." \
                     %(bc.__class__.__name__))
        return bcu

    def Transient_update(self):
        # Update to new time-level
        if 'velocity_update' in self.pdesubsystems: 
            dummy = self.pdesubsystems['velocity_update'].solve()
        NSSolver.Transient_update(self)
        
######### Factory functions or classes for numerical pdesubsystems ###################
 
class VelocityBase(PDESubSystem):
    """ Velocity update using constant mass matrix """
        
    def define(self):
        form_args = self.solver_namespace.copy()
        # Pressure in the velocity equation? False for chorin
        self.prm['include_pressure'] = True 
        self.exterior = any([bc.type() in ['ConstantPressure', 'Outlet'] 
                            for bc in self.bcs])
        self.Laplace_U = self.solver_namespace['u'] # Used in add_exterior.
        if self.prm['iteration_type'] == 'Picard':
            self.get_form(form_args)
            if self.F:
                # Add contribution from boundaries
                if self.exterior: 
                    self.F = self.F + self.add_exterior(**form_args)
                self.a, self.L = lhs(self.F), rhs(self.F)
        else:
            form_args['u_'] = self.solver_namespace['u']
            self.get_form(form_args)
            if self.F:
                # Add contribution from boundaries
                if self.exterior: 
                    self.F = self.F + self.add_exterior(**form_args)
                # Set up Newton system
                u_, u = self.solver_namespace['u_'], self.solver_namespace['u']
                F_ = action(self.F, function = u_)
                J = derivative(F_, u_, u)
                self.a, self.L = J, -F_
            
    def add_exterior(self, u, v, p_, n, nu, **kwargs):
        """
        Set the weak boundary condition.
        This boundary condition should be added to the Navier-Stokes form.
        """
        u = self.Laplace_U
        L = []
        for bc in self.bcs:
            if bc.type() == 'ConstantPressure':
                info('Assigning weak boundary condition for ' + bc.type())
                L.append(-nu*inner(v, grad(u).T*n)*ds(bc.bid))
                if self.prm['include_pressure']: 
                    L.append(inner(p_*n, v)*ds(bc.bid))
                self.exterior_facet_domains = bc.mf 
                
            elif bc.type() == 'Symmetry':
                info('Assigning weak boundary condition for ' + bc.type())
                L.append(-nu*inner(v, grad(u).T*n)*ds(bc.bid) 
                         + inner(v[0], inner(u, n))*ds(bc.bid))
                if self.prm['include_pressure']: 
                   L.append(inner(p_*n, v)*ds(bc.bid))
                self.exterior_facet_domains = bc.mf         
                
            elif bc.type() == 'Outlet':
                info('Assigning weak boundary condition for ' + bc.type())
                L.append(-nu*inner(v, grad(u).T*n)*ds(bc.bid))
                if self.prm['include_pressure']: 
                   L.append(inner(p_*n, v)*ds(bc.bid))
                self.exterior_facet_domains = bc.mf

        return reduce(operator.add, L)
        
class VelocityUpdateBase(PDESubSystem):
    """ Velocity update using constant mass matrix """
    def __init__(self, solver_namespace, unknown, bcs=[], **kwargs):
        PDESubSystem.__init__(self, solver_namespace, unknown, 
                                    bcs=bcs, **kwargs)
                                    
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
        
        self.solver_namespace['dp_'] = Function(self.V)
        self.solver_namespace['dpx'] = self.solver_namespace['dp_'].vector()
        self.p_old = Function(self.V)
        self.px_old = self.p_old.vector()
        self.prm['iteration_type'] = 'Picard'
        
    def define(self):
        self.prm['pressure_correction'] = True
        if any([bc.type() in ['ConstantPressure', 'Outlet'] for bc in self.bcs]): 
            self.normalize=None
        self.get_form(self.solver_namespace)            
        self.exterior = self.add_exterior(**self.solver_namespace)

        if self.F: 
            if self.exterior: 
                self.F = self.F + self.exterior
            self.a, self.L = lhs(self.F), rhs(self.F)

    def add_exterior(self, p, p_, q, n, dt, u_, nu, **kwargs):        
        L = []
        for bc in self.bcs:
            if bc.type() == 'VelocityInlet':
                #if u_.ufl_element().degree() < 2:
                #    error('Requires order > 1 ')
                if not self.prm['pressure_correction']:
                    # inner(q*n, grad(p))*ds is significant and necessary for chorin.
                    # For pressure correction pdesubsystems the term can be neglected because 
                    # it then equals inner(q*n, grad(dp_))*ds
                    #L.append(-inner(q*n, nu*div(grad(u_)))*ds(bc.bid))
                    L.append(-inner(q*n, grad(p_))*ds(bc.bid))
                
        if len(L) > 0:
            return reduce(operator.add, L)
        else:
            return False
                
    def prepare(self):
        """ Remember old pressure solution """
        self.px_old[:] = self.x[:]
        
    def update(self):
        """ Get pressure correction """
        self.solver_namespace['dpx'][:] = self.x[:] - self.px_old[:]
        
    def Transient_update(self):
        pass    

############# Velocity update #################################
class VelocityUpdate_1(VelocityUpdateBase):
    """ Velocity update using constant mass matrix """
    def form(self, u_, u, v, dt, dp_, p_, **kwargs):
        # No need to reassemble the mass matrix
        self.prm['reassemble_lhs'] = False 
        F = inner(u - u_, v)*dx
        if self.solver_namespace['pdesubsystems']['u'].prm['include_pressure']:
            F = F + inner(dt*grad(dp_), v)*dx
        else:
            F = F + inner(dt*grad(p_), v)*dx
        return F

############# Velocity update #################################

############# Pressure ########################################
class Transient_Pressure_1(PressureBase):
    
    def form(self, p_, p, q, u_, dt, **kwargs):   
        self.prm['reassemble_lhs'] = False
        F = inner(grad(q), grad(p))*dx
        if self.solver_namespace['pdesubsystems']['u'].prm['include_pressure']:
            F = F - inner(grad(q), grad(p_))*dx + (1./dt)*q*div(u_)*dx
        else:
            self.prm['pressure_correction'] = False
            F = F + (1./dt)*q*div(u_)*dx
        return F

class Steady_Pressure_1(PressureBase):
    
    def form(self, p_, p, q, u_, dt, convection_form, **kwargs):
        self.prm['reassemble_lhs'] = False
        self.prm['pressure_correction'] = False
        return inner(grad(q), grad(p))*dx + \
               self.conv(grad(q), u_, u_, convection_form)*dx  + \
               q*div(u_)*dx
        
class Steady_Pressure_2(PressureBase):
    
    def form(self, p_, p, q, u_, dt, convection_form, **kwargs):
        self.prm['reassemble_lhs'] = False
        self.prm['pressure_correction'] = False
        return inner(grad(q), grad(p))*dx + \
               self.conv(grad(q), u_, u_, convection_form)*dx
        
class Steady_Pressure_3(PressureBase):
    
    def form(self, p_, p, q, u_, dt, **kwargs):   
        self.prm['reassemble_lhs'] = False
        F = inner(grad(q), grad(p))*dx
        if self.solver_namespace['pdesubsystems']['u'].prm['include_pressure']:
            F = F - inner(grad(q), grad(p_))*dx + (1./dt)*q*div(u_)*dx
        else:
            F = F + (1./dt)*q*div(u_)*dx
        return F

############# Pressure ########################################

############# Velocity ########################################
class Transient_Velocity_1(VelocityBase):
    """Incremental pressure correction.
    Crank-Nicholson (CN) diffusion. Convection is computed using 
    AB-projection for convecting and CN for convected velocity.
    """       
    def form(self, u_, u, v, p_, u_1, u_2, nu, nut_, f, dt, convection_form,
             **kwargs): 
        U_ = 1.5*u_1 - 0.5*u_2
        U  = 0.5*(u + u_1)
        self.Laplace_U = U
        return (1./dt)*inner(u - u_1, v)*dx + \
               self.conv(v, U, U_, convection_form)*dx + \
               nu*inner(grad(U) + grad(U).T, grad(v))*dx - \
               inner(div(v), p_)*dx - inner(v, f)*dx

class Transient_Velocity_2(VelocityBase):
    """Incremental pressure correction.
    Crank-Nicholson (CN) diffusion. Convection is computed with explicit 
    Adams-Bashforth projection and the coefficient matrix can be preassembled.
    Scheme is linear and Newton returns the same as Picard.
    """
    def form(self, u_, u, v, p_, u_1, u_2, nu, nut_, f, dt, convection_form, 
             **kwargs):    
        if type(nu) is Constant and self.prm['iteration_type'] == 'Picard':
            self.prm['reassemble_lhs'] = False
        U  = 0.5*(u + u_1)
        self.Laplace_U = U
        return (1./dt)*inner(u - u_1, v)*dx + \
               1.5*self.conv(v, u_1, u_1, convection_form)*dx - \
               0.5*self.conv(v, u_2, u_2, convection_form)*dx + \
               nu*inner(grad(U) + grad(U).T, grad(v))*dx - \
               inner(div(v), p_)*dx - inner(v,f)*dx

class Transient_Velocity_3(VelocityBase):
    """Chorin. 
    Crank-Nicholson (CN) diffusion. Convection is computed using 
    AB-projection for convecting and CN for convected velocity.
    Note, pdesubsystem is non-iterative, because the velocity equation does
    not include pressure. Use max_iter=1.
    """
    def form(self, u_, u, v, p_, u_1, u_2, nu, nut_, f, dt, 
             convection_form, **kwargs): 
        U_ = 1.5*u_1 - 0.5*u_2
        U  = 0.5*(u + u_1)
        self.Laplace_U = U
        self.prm['include_pressure'] = False
        return (1./dt)*inner(u - u_1, v)*dx + \
               self.conv(v, U, U_, convection_form)*dx + \
               nu*inner(grad(U) + grad(U).T, grad(v))*dx - inner(v, f)*dx

class Transient_Velocity_4(VelocityBase):
    """Incremental pressure correction.
    Crank-Nicholson (CN) diffusion. Convection is computed with a
    central pdesubsystem that requires iterating over u_.
    """
    def form(self, u_, u, v, p_, u_1, u_2, p_1, p_2, nu, nut_, f, dt, 
             convection_form, **kwargs): 
        U_ = 0.5*(u_ + u_1)
        U  = 0.5*(u + u_1)
        self.Laplace_U = U
        self.prm['max_inner_iter'] = 3 # Depends on u_
        return (1./dt)*inner(u - u_1, v)*dx + \
               self.conv(v, U, U_, convection_form)*dx + \
               nu*inner(grad(U) + grad(U).T, grad(v))*dx - \
               inner(div(v), p_)*dx - inner(v, f)*dx
               
class Transient_Velocity_5(VelocityBase):
    """Incremental pressure correction.
    Crank-Nicholson (CN) diffusion. Convection is computed with a
    central pdesubsystem that requires iterating over u_.
    More suitable for Newton iterations.
    """       
    def form(self, u_, u, v, p_, u_1, u_2, p_1, p_2, nu, nut_, f, dt, 
             convection_form, **kwargs): 
        U  = 0.5*(u + u_1)
        self.Laplace_U = U
        self.prm['max_inner_iter'] = 3 # Depends on u_
        return (1./dt)*inner(u - u_1, v)*dx + \
               0.5*self.conv(v, u_, u_, convection_form)*dx + \
               0.5*self.conv(v, u_1, u_1, convection_form)*dx + \
               nu*inner(grad(U) + grad(U).T, grad(v))*dx - \
               inner(div(v), p_)*dx - inner(v, f)*dx

class Transient_Velocity_10(VelocityBase):
    """Incremental pressure correction.
    Crank-Nicholson (CN) diffusion. Convection is computed using 
    AB-projection for convecting and CN for convected velocity.
    """
    def form(self, u_, u, v, p_, u_1, u_2, nu, nut_, f, dt, convection_form, 
             **kwargs):    
        if self.exterior:
            info_red('Warning!! Does not work with given boundaries because of non-symmetric Laplacian.')
            
        U_ = 1.5*u_1 - 0.5*u_2
        U  = 0.5*(u + u_1)
        self.Laplace_U = U
        return (1./dt)*inner(u - u_1, v)*dx + \
               self.conv(v, U, U_, convection_form)*dx + \
               nu*inner(grad(U), grad(v))*dx + \
               inner(v, grad(p_))*dx - inner(v, f)*dx

class Transient_Velocity_20(VelocityBase):
    """Incremental pressure correction.
    Crank-Nicholson (CN) diffusion. Convection is computed with explicit 
    Adams-Bashforth projection and the coefficient matrix can be preassembled.
    Scheme is linear and Newton returns the same as Picard.
    """
    def form(self, u_, u, v, p_, u_1, u_2, nu, nut_, f, dt, convection_form, 
             **kwargs):    
        if self.exterior:
            info_red('Warning!! Does not work with given boundaries because of non-symmetric Laplacian.')
        if type(nu) is Constant and self.prm['iteration_type'] == 'Picard':
            self.prm['reassemble_lhs'] = False
        U  = 0.5*(u + u_1)
        self.Laplace_U = U
        return (1./dt)*inner(u - u_1, v)*dx + \
               self.conv(v, u_1, u_1, convection_form)*dx - \
               nu*inner(grad(U), grad(v))*dx + \
               inner(v, grad(p_))*dx - inner(v,f)*dx

class Transient_Velocity_30(VelocityBase):
    """GRPC
    """       
    def form(self, u_, u, v, p_, u_1, u_2, nu, nut_, f, dt, convection_form,
             **kwargs): 
        U_  = 0.5*(u_ + u_1)
        self.Laplace_U = 0.5*dt(0)*u - dt(0)*U_
        self.prm['reassemble_lhs'] = False
        Ru = inner(u_ - u_1, v)*dx + \
             dt*self.conv(v, U_, U_, convection_form)*dx + \
             dt*inner(epsilon(v), sigma(U_, p_, nu))*dx - \
             dt*inner(v, f)*dx
             
        ax = inner(v, u)*dx + 0.5*dt*2*nu*inner(epsilon(v), epsilon(u))*dx
             
        return ax - Ru
        
    def solve_Picard_system(self, assemble_A, assemble_b):
        """One assemble and solve of system.
        The GRPC is a nonlinear type solver that updates the solution vector with a residual dx.
        """
        if assemble_A: self.assemble(self.A)
        if assemble_b: self.assemble(self.b)
        if assemble_A:
            [bc.apply(self.A) for bc in self.bcs]
        if assemble_b:
            [bc.apply(self.b, self.x) for bc in self.bcs]
        dx = self.work   # more informative name
        dx.zero()        # start vector for iterative solvers
        self.linear_solver.solve(self.A, dx, self.b)
        self.x.axpy(-1.0, dx)  # relax
        return norm(self.b), dx
        
    def add_exterior(self, u, v, u_, u_1, p_, n, nu, dt, **kwargs):
        """
        Set the weak boundary condition.
        This boundary condition should be added to the Navier-Stokes form.
        """
        u = self.Laplace_U
        L = []
        for bc in self.bcs:
            if bc.type() == 'ConstantPressure':
                info('Assigning weak boundary condition for ' + bc.type())
                L.append(-nu*inner(v, grad(u).T*n)*ds(bc.bid))
                L.append(-dt*inner(bc.func['p']*n, v)*ds(bc.bid))
                self.exterior_facet_domains = bc.mf 
                
            elif bc.type() == 'Symmetry':
                info('Assigning weak boundary condition for ' + bc.type())
                L.append(-nu*inner(v, grad(u).T*n)*ds(bc.bid) 
                         + inner(v[0], inner(u, n))*ds(bc.bid))
                L.append(dt*inner(p_*n, v)*ds(bc.bid))
                self.exterior_facet_domains = bc.mf         
                
            elif bc.type() == 'Outlet':
                info('Assigning weak boundary condition for ' + bc.type())
                L.append(-nu*inner(v, grad(u).T*n)*ds(bc.bid))
                L.append(dt*inner(p_*n, v)*ds(bc.bid))
                self.exterior_facet_domains = bc.mf

        return reduce(operator.add, L)
        
class Transient_Pressure_30(PressureBase):
    
    def form(self, p_, p, q, u_, u_1,nu, dt, **kwargs):   
        self.prm['reassemble_lhs'] = False
        F = dt**2*(inner(grad(q), grad(p)) + (1.0/(nu*dt))*q*(p))*dx \
            - dt*q*div(0.5*(u_+u_1))*dx
        return F        
        
    def solve_Picard_system(self, assemble_A, assemble_b):
        """One assemble and solve of system.
        The GRPC is a nonlinear type solver that updates the solution vector with a residual dx.
        """
        if assemble_A: self.assemble(self.A)
        if assemble_b: self.assemble(self.b)
        if assemble_A:
            [bc.apply(self.A, self.x) for bc in self.bcs]            
        if assemble_b:
            [bc.apply(self.b, self.x) for bc in self.bcs]
        dx = self.work   # more informative name
        dx.zero()        # start vector for iterative solvers
        self.linear_solver.solve(self.A, dx, self.b)
        if self.normalize: self.normalize(dx)
        self.x.axpy(-2.0, dx)  # relax
        #self.x[:] = dx[:]
        return norm(self.b), dx

##  Note! Steady (not pseudo-transient) segregated solvers need 
##  underrelaxation!

class Steady_Velocity_1(VelocityBase):
    
    def form(self, u_, u, v, p_, nu, nut_, f, dt, convection_form, **kwargs):
        return self.conv(v, u, u_, convection_form)*dx + \
               nu*inner(grad(u) + grad(u).T, grad(v))*dx - \
               inner(div(v), p_)*dx - inner(v, f)*dx

class Steady_Velocity_2(VelocityBase):
    
    def form(self, u_, u, v, p_, nu, nut_, f, dt, convection_form, **kwargs):
        self.prm['reassemble_lhs'] = False        
        return self.conv(v, u_, u_, convection_form)*dx + \
               nu*inner(grad(u) + grad(u).T, grad(v))*dx - \
               inner(div(v), p_)*dx - inner(v,f)*dx

class Steady_Velocity_3(VelocityBase):
    """ Pseudo-transient steady solver"""        
    def form(self, u_, u, v, p_, nu, nut_, f, dt, convection_form, **kwargs): 
        self.prm['reassemble_lhs'] = False    
        if abs(self.prm['omega'] - 1) > 2.*DOLFIN_EPS: 
            info_red('Warning! Pseudo-transient pdesubsystem should probably be \
                      used without underrelaxation')
        return (1./dt)*inner(v, u - u_)*dx + \
               self.conv(v, u_, u_, convection_form)*dx + \
               nu*inner(grad(u) + grad(u).T, grad(v))*dx - \
               inner(div(v), p_)*dx - inner(v,f)*dx 
            
############# Velocity  #########################################

from NSSegregated_optimized import *
