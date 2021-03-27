__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-08-30"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"

from NSSolver import *

class NSCoupled(NSSolver):
   
    def __init__(self, problem, parameters):
        NSSolver.__init__(self, system_composition=[['u', 'p']],
                                problem=problem, 
                                parameters=parameters)   
                            
    def define(self):
        # Create normalization function for pressure 
        self.normalize['up'] = extended_normalize(self.V['up'], self.u_.value_size())
        classname = self.prm['time_integration'] + '_Coupled_' + \
                    str(self.prm['pdesubsystem']['up'])
        self.pdesubsystems['up'] = eval(classname)(vars(self), ['u', 'p'],
                                             bcs=self.bc['up'], 
                                             normalize=self.normalize['up'])

    def create_BCs(self, bcs):
        """
        Create boundary conditions for velocity and pressure
        based on boundaries in list bcs.
        """
        bcu = {'up': []}
        for bc in bcs:
            if bc.type() in ('VelocityInlet', 'Wall'):
                if hasattr(bc, 'func'): # Use func if provided
                    add_BC(bcu['up'], self.V['up'].sub(0), bc, bc.func['u'])
                else:
                    if bc.type() == 'Wall':
                        # Default is zero on walls for all quantities
                        func = Constant((1e-10,)*self.mesh.geometry().dim())
                        add_BC(bcu['up'], self.V['up'].sub(0), bc, func)                        
                    elif bc.type() == 'VelocityInlet':
                        raise TypeError, 'expected func for VelocityInlet'                
            elif bc.type() in ('ConstantPressure', 'Outlet', 'Symmetry', 'Slip'):
                # This bc is weakly enforced
                bcu['up'].append(bc)
            elif bc.type() == 'Periodic':
                add_BC(bcu['up'], self.V['up'], bc, None)
            else:
                info("No assigned boundary condition for %s -- skipping..."      
                     %(bc.__class__.__name__))
        return bcu

class CoupledBase(PDESubSystem):
    """Base class for pdesubsystems of the coupled solver."""
        
    def define(self):
        
        form_args = self.solver_namespace.copy()
        # Check if there are boundaries that require a weak ds form
        self.exterior = any([bc.type() in ['ConstantPressure', 'Outlet']
                             for bc in self.bcs]) 
        self.Laplace_U = self.solver_namespace['u'] # Velocity used in add_exterior. 
        if self.prm['iteration_type'] == 'Picard':
            self.get_form(form_args)
            if self.F:
                if self.exterior: 
                    self.F = self.F + self.add_exterior(**form_args)
                # Add stabilization if function spaces are equal
                if (self.V.sub(0).ufl_element().degree() ==
                    self.V.sub(1).ufl_element().degree()): 
                    self.F = self.F + self.stabilization(**form_args)                    
                    
                self.a, self.L = lhs(self.F), rhs(self.F)
        else:
            form_args['u_'] = self.solver_namespace['u']
            form_args['p_'] = self.solver_namespace['p']
            #self.Laplace_U = self.solver_namespace['u_'] # Velocity used in add_exterior. 
            self.get_form(form_args)
            if self.F:
                # Add contribution from boundaries
                if self.exterior: 
                    self.F = self.F + self.add_exterior(**form_args)
                # Add stabilization if functionspaces are equal
                if (self.V.sub(0).ufl_element().degree() ==
                    self.V.sub(1).ufl_element().degree()):
                    self.F = self.F + self.stabilization(**form_args) 
                # Set up Newton system
                up_, up = self.solver_namespace['up_'], self.solver_namespace['up']
                F_ = action(self.F, coefficient=up_)
                J_ = derivative(F_, up_, up)
                self.a, self.L = J_, -F_

    def add_exterior(self, u, v, p, q, n, nu, **kwargs):
        """
        Set the weak boundary condition for pressure using a provided func.
        This boundary condition should be added to the Navier-Stokes form.
        """
        U = self.Laplace_U  # Get the velocity used in Laplace form
        L = []
        for bc in self.bcs:
            if bc.type() == 'ConstantPressure':
                info_green('Assigning weak boundary condition for ' + bc.type())
                L.append(-nu*inner(v, grad(U).T*n)*ds(bc.bid) +
                         inner(bc.func['p']*n, v)*ds(bc.bid)) 
                self.exterior_facet_domains = bc.mf 
                self.normalize = None                    
                
            elif bc.type() in ('Outlet'):
                info_green('Assigning psuedo traction boundary condition for ' + bc.type())
                L.append(-nu*inner(v, grad(U).T*n)*ds(bc.bid))
                self.exterior_facet_domains = bc.mf         
                
            elif bc.type() in ('Symmetry'):
                # Correct way to handle Symmetry?
                info_green('Assigning weak boundary condition for ' + bc.type())
                L.append(-nu*inner(v, grad(U).T*n)*ds(bc.bid) + 
                         inner(p*n, v)*ds(bc.bid) + inner(q*n, U)*ds(bc.bid))
                self.exterior_facet_domains = bc.mf      
                
            elif bc.type() in ('Slip'):
                info_green('Assigning slip boundary condition for ' + bc.type())
                L.append(-nu*inner(v, grad(U)*n + grad(U).T*n)*ds(bc.bid) + 
                         inner(p*n, v)*ds(bc.bid) + inner(q*n, U)*ds(bc.bid))
                self.exterior_facet_domains = bc.mf      
                                                
        return reduce(operator.add, L)
        
    def Steady_NS_form(self, u, v, p, q, u_, nu, f, convection_form):
        """Steady Navier-Stokes form."""
        F = self.conv(v, u, u_, convection_form)*dx \
            + nu*inner(grad(v), grad(u) + grad(u).T)*dx - inner(div(v), p)*dx \
            - inner(q, div(u))*dx - inner(v, f)*dx
        return F
        
    def stabilization(self, u_, u, p, q, nu, nut_, eps, n, convection_form, 
                      **kwargs):
        """Add stabilization to Navier Stokes solver."""
        if type(nu) is Constant:
            info_green('Adding stabilization for constant nu')
            F = - eps*inner(grad(q), grad(p))*dx \
                - eps*inner(grad(q), dot(grad(u), u_))*dx
            for bc in self.bcs:
                if bc.type() in ('ConstantPressure', 'Outlet', 
                                 'VelocityInlet'):
                    F = F + eps*inner(q, dot(grad(p), n))*ds(bc.bid)
        else:
            info_green('Adding stabilization for variable nu')
            F = - eps*inner(grad(q), grad(p))*dx \
                - eps*inner(grad(q), dot(grad(u), u_))*dx \
                - eps*inner(grad(grad(q)*nut_), grad(u))*dx \
                + eps*inner(outer(grad(q), grad(nut_)), grad(u) + grad(u).T)*dx
            for bc in self.bcs:
                if bc.type() in ('ConstantPressure', 'Outlet', 
                                 'VelocityInlet'):
                    F = F + eps*inner(q, dot(grad(p), n))*ds(bc.bid)        
        return F
        
    def get_precond(self, u, v_u, p, v_p, f, nu, **kwargs):
        # Form for use in constructing preconditioner matrix
        b = 0.5*nu*inner(grad(u) + grad(u).T, grad(v_u))*dx - p*v_p*dx
        
        # Assemble preconditioner system. B is hooked up in setup_solver
        if self.prm['reassemble_precond']: # True first time around
            self.B = assemble(b, tensor=self.B, exterior_facet_domains=self.exterior_facet_domains)
            [bc.apply(self.B) for bc in self.bcs]
            self.prm['reassemble_precond'] = False # Constant form
        return self.B
        
class Steady_Coupled_1(CoupledBase):
    """Simplest possible steady form."""
    def form(self, u, v, p, q, u_, nu, f, convection_form, **kwargs):
        return self.Steady_NS_form(u, v, p, q, u_, nu, f, convection_form)

class Steady_Coupled_2(CoupledBase):
    """Use explicit convection and preassemble the coefficient matrix for 
    Picard iterations. For Newton it's still nonlinear and needs reassembling.
    """
    def form(self, u, v, p, q, u_, nu, f, convection_form, **kwargs):
        if self.prm['iteration_type'] == 'Picard' and type(nu) is Constant:
            self.prm['reassemble_lhs'] = False
        F = self.conv(v, u_, u_, convection_form)*dx \
            + nu*inner(grad(v), grad(u) + grad(u).T)*dx - inner(div(v), p)*dx \
            - inner(q, div(u))*dx - inner(v, f)*dx
        return F

class Steady_Coupled_3(CoupledBase):
    """Pseudo-Transient form. For Newton the transient term is zero though."""
    def form(self, u, v, p, q, u_, nu, f, convection_form, dt, **kwargs):
        return (1./dt)*inner(u - u_,v)*dx \
               + self.Steady_NS_form(u, v, p, q, u_, nu, f, convection_form)

class Steady_Coupled_4(CoupledBase):
    """Simple steady form to illustrate that one can preassemble parts of the
    system.
    """
    def form(self, u, v, p, q, u_, nu, f, convection_form, **kwargs):
        if type(nu) is Constant:
            F = self.conv(v, u, u_, convection_form)*dx
            F_pre = nu*inner(grad(v), grad(u) + grad(u).T)*dx - \
                    inner(div(v), p)*dx - inner(q, div(u))*dx - inner(v, f)*dx
        else:
            F = self.conv(v, u, u_, convection_form)*dx \
                + nu*inner(grad(v), grad(u) + grad(u).T)*dx
            F_pre = - inner(div(v), p)*dx - inner(q, div(u))*dx - \
                      inner(v, f)*dx
           
        if self.prm['iteration_type'] == 'Picard':
            a, L = lhs(F_pre), rhs(F_pre)    
            # Hold the preassembled tensors
            self.A1 = assemble(a, tensor = self.A1,
                            exterior_facet_domains=self.exterior_facet_domains)
            self.b1 = assemble(L, tensor = self.b1,
                            exterior_facet_domains=self.exterior_facet_domains)
            return F
        else:
            return F + F_pre

class Steady_Coupled_5(CoupledBase):
    """Coupled form for use with Reynolds stress model."""
    def form(self, u, v, p, q, u_, nu, f, convection_form, correction, **kwargs):
        if correction:
            return self.Steady_NS_form(u, v, p, q, u_, nu, f, convection_form) - inner(grad(v), correction)*dx
        else:
            return self.Steady_NS_form(u, v, p, q, u_, nu, f, convection_form)

class Transient_Coupled_1(CoupledBase):
    """Transient form with Crank-Nicholson (CN) diffusion and where convection 
    is computed using AB-projection for convecting and CN for convected 
    velocity.
    """
    def form(self, u, v, p, q, u_, nu, f, convection_form, dt, u_1, u_2, 
             **kwargs):
        U = 0.5*(u + u_1)
        U1 = 1.5*u_1 - 0.5*u_2
        self.Laplace_U = U
        F = (1./dt)*inner(u - u_1, v)*dx + \
            self.conv(v, U, U1, convection_form)*dx + \
            nu*inner(grad(v), grad(U) + grad(U).T)*dx - inner(div(v), p)*dx - \
            inner(q, div(u))*dx - inner(v, f)*dx
        return F

class Transient_Coupled_2(CoupledBase):
    """Transient form with Crank-Nicholson diffusion and where convection is 
    computed iteratively. Scheme requires inner iterations because it uses the 
    latest approximation to the solution, which is u_.
    """
    def form(self, u, v, p, q, u_, nu, f, convection_form, dt, u_1, u_2, 
             **kwargs):
        U = 0.5*(u + u_1)
        U_ = 0.5*(u_ + u_1)
        self.Laplace_U = U
        F = (1./dt)*inner(u - u_1, v)*dx + \
            self.conv(v, U, U_, convection_form)*dx + \
            nu*inner(grad(v), grad(U) + grad(U).T)*dx - inner(div(v), p)*dx - \
            inner(q, div(u))*dx - inner(v, f)*dx
        self.prm['max_inner_iter'] = 3 # Depends on u_
        return F

class Transient_Coupled_3(CoupledBase):
    """Transient form with Crank-Nicholson diffusion and where convection is 
    computed iteratively. Scheme requires inner iterations because it uses the 
    latest approximation to the solution, which is u_. If iteration type is 
    Picard, then we preassemble everything not changing in time.
    """
    def form(self, u, v, p, q, u_, nu, f, convection_form, dt, u_1, u_2, 
             **kwargs):
        U = 0.5*(u + u_1)
        U_ = 0.5*(u_ + u_1)
        self.Laplace_U = U
        F = (1./dt)*inner(u - u_1, v)*dx + \
            self.conv(v, U, U_, convection_form)*dx + \
            0.5*nu*inner(grad(v), grad(u_1) + grad(u_1).T)*dx       
        F_pre = - inner(div(v), p)*dx - inner(q, div(u))*dx - inner(v, f)*dx
        if type(nu) is Constant:
            F_pre = F_pre + 0.5*nu*inner(grad(v), grad(u) + grad(u).T)*dx
        else:
            F = F + 0.5*nu*inner(grad(v), grad(u) + grad(u).T)*dx
        self.prm['max_inner_iter'] = 3    # Depends on u_
        if self.prm['iteration_type'] == 'Picard':
            a, L = lhs(F_pre), rhs(F_pre)
            # Hold the preassembled tensors
            self.A1 = assemble(a, tensor=self.A1,
                           exterior_facet_domains=self.exterior_facet_domains) 
            self.b1 = assemble(L, tensor=self.b1,
                           exterior_facet_domains=self.exterior_facet_domains)
            return F
        else:
            return F + F_pre

class Transient_Coupled_4(CoupledBase):
    """Transient form with Crank-Nicholson diffusion and where convection is 
    computed with explicit Adams-Bashforth projection and the coefficient 
    matrix can be preassembled. Scheme is linear and Newton returns the same as 
    Picard.
    """
    def form(self, u, v, p, q, u_, nu, f, convection_form, dt, u_1, u_2, 
             **kwargs):
        if self.prm['iteration_type'] == 'Picard' and type(nu) is Constant:
            self.prm['reassemble_lhs'] = False
        U = 0.5*(u + u_1)
        self.Laplace_U = U
        F = (1./dt)*inner(u-u_1, v)*dx + \
            1.5*self.conv(v, u_1, u_1, convection_form)*dx - \
            0.5*self.conv(v, u_2, u_2, convection_form)*dx + \
            nu*inner(grad(v), grad(U) + grad(U).T)*dx - inner(div(v), p)*dx - \
            inner(q, div(u))*dx - inner(v, f)*dx
        return F

class Transient_Coupled_5(CoupledBase):
    """Transient form with Crank-Nicholson diffusion and where convection is 
    computed iteratively. Scheme requires inner iterations because it uses the 
    latest approximation to the solution, which is u_.
    """
    def form(self, u, v, p, q, u_, nu, f, convection_form, dt, u_1, u_2, 
             **kwargs):
        U = 0.5*(u + u_1)
        self.Laplace_U = U
        F = (1./dt)*inner(u - u_1, v)*dx + \
            0.5*self.conv(v, u, u_, convection_form)*dx + \
            0.5*self.conv(v, u_1, u_1, convection_form)*dx + \
            nu*inner(grad(v), grad(U) + grad(U).T)*dx - inner(div(v), p)*dx - \
            inner(q, div(u))*dx - inner(v, f)*dx
        self.prm['max_inner_iter'] = 3 # Depends on u_
        return F
        
#class Transient_Coupled_6(CoupledBase):
    #"""Transient form with Crank-Nicholson diffusion and where convection is 
    #computed with explicit 3rd order Adams-Bashforth projection and the
    #coefficient matrix can be preassembled. Scheme is linear and Newton returns
    #the same as Picard.
    #"""
    #def form(self, u, v, p, q, u_, nu, f, convection_form, dt, u_1, u_2, 
             #**kwargs):
        #if self.prm['iteration_type'] == 'Picard' and type(nu) is Constant:
            #self.prm['reassemble_lhs'] = False
        #U = 0.5*(u + u_1)
        #self.Laplace_U = U
        #self.solver.add_function_on_timestep(['3'])
        #u_3 = self.solver.u_3
        #F = (1./dt)*inner(u - u_1, v)*dx + \
            #1./12.*(23.*self.conv(v, u_1, u_1, convection_form)*dx - \
                    #16.*self.conv(v, u_2, u_2, convection_form)*dx + \
                    #5. *self.conv(v, u_3, u_3, convection_form)*dx) + \
            #nu*inner(grad(v), grad(U) + grad(U).T)*dx - inner(div(v), p)*dx - \
            #inner(q, div(u))*dx - inner(v, f)*dx
        #return F

#class Transient_Coupled_7(CoupledBase):
    #"""Transient form with 4th order Adams-Moulton diffusion and where 
    #convection is computed with explicit 3rd order Adams-Bashforth projection
    #and the coefficient matrix can be preassembled. Scheme is linear and Newton
    #returns the same as Picard.
    #"""
    #def form(self, u, v, p, q, u_, nu, f, convection_form, dt, u_1, u_2, 
             #**kwargs):
        #if self.prm['iteration_type'] == 'Picard' and type(nu) is Constant:
            #self.prm['reassemble_lhs'] = False
        #self.solver['add_function_on_timestep'](['3'])
        #u_3 = self.solver['u_3']
        #U = 1./24.*(9.*u + 19.*u_1 - 5.*u_2 + u_3)
        #self.Laplace_U = U
        #F = (1./6./dt)*inner(11.*u - 18.*u_1 + 9.*u_2 -2.*u_3, v)*dx + \
            #1./12.*(23.*self.conv(v, u_1, u_1, convection_form)*dx - \
                    #16.*self.conv(v, u_2, u_2, convection_form)*dx + \
                    #5. *self.conv(v, u_3, u_3, convection_form)*dx) +\
            #nu*inner(grad(v), grad(U) + grad(U).T)*dx - inner(div(v), p)*dx - \
            #inner(q, div(u))*dx - inner(v, f)*dx
        #return F
        
#from NSCoupled_optimized import *
