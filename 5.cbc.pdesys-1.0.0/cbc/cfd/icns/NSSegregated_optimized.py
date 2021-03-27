__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-08-30"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"

##############################################################################
############################ Optimized schemes ###############################
##############################################################################

from NSSegregated import *

class Transient_Velocity_101(VelocityBase):
    """ 
    Optimized solver for variable or constant nu.
    In this solver we preassemble everything but the convection (and diffusion 
    if nu is not Constant). The right hand side is computed using fast matrix 
    vector multiplications. The Convection uses Crank-Nicolson for convected 
    and Adams-Bashforth for convecting velocity. The diffusion uses 
    Crank-Nicolson.
    """ 
    
    def form(self, u_, u, v, p, q, p_, u_1, u_2, nu, nut_, f, dt, convection_form,
             nuM, x_1, **kwargs): 
             
        U1 = 1.5*u_1 - 0.5*u_2       # AB-projection
        if type(nu) is Constant:
            self.nu_ = nu_ = nuM
        else:
            raise NotImplementedError
            #self.nu_ = nu_ = nuM + 0.5*(nut_ + nut_1)
        aM = inner(v, u)*dx       # Segregated Mass matrix
        aP = inner(v, grad(p))*dx   #
        if type(nu_) is Constant:
            self.a = 0.5*self.conv(v, u, U1, convection_form)*dx
            self.aK = nu_*inner(grad(v), grad(u))*dx
            self.K = assemble(self.aK)
        else:
            raise NotImplementedError
            #self.a = 0.5*self.conv(v1, u1, U1, convection_form)*dx + 0.5*nu_*inner(grad(v1), grad(u1))*dx
            #self.K = None
        self.L = None
        self.dt = dt
        self.x_1 = x_1['u']
        self.f0 = interpolate(f, self.V)
        # Assemble matrices that don't change
        self.M = assemble(aM)
        self.P = assemble(aP)
        # Set the initial rhs-vector equal to the constant body force.
        self.b = Vector(self.x)
        self.bold = Vector(self.x)
        self.b0 = self.M*self.f0.vector()
        self.prm['reassemble_lhs_inner'] = False
        self.exterior = False
        return False
            
    def assemble(self, dummy):
        self.b[:] = self.b0[:] # body force
        # A contains convection if nu is Constant and convection + diffusion otherwise
        self.A = assemble(self.a, tensor=self.A, reset_sparsity=self.prm['reset_sparsity']) 
        self.A._scale(-1.) # Negative convection and diffusion on the rhs 
        self.A.axpy(1./self.dt(0), self.M, True) # Add mass
        if not self.K is None:
            self.A.axpy(-0.5, self.K, True) # add diffusion if missing
        self.b.axpy(1., self.A*self.x_1)
        # Reset matrix for lhs
        self.A._scale(-1.)
        self.A.axpy(2./self.dt(0), self.M, True)
        self.prm['reset_sparsity'] = False
        
    def solve_Picard_system(self, assemble_A, assemble_b):
        self.prepare()
        if assemble_A: 
            self.assemble(None)
            [bc.apply(self.A) for bc in self.bcs]
        self.bold[:] = self.b[:] # This part is not reassembled on inner iters, so remember it
        self.b.axpy(-1., self.P*self.solver_namespace['x_']['p'])
        [bc.apply(self.b) for bc in self.bcs]
        x_star = self.work  # more informative name
        x_star[:] = self.x[:]    # start vector for iterative solvers
        rv = residual(self.A, x_star, self.b)
        self.setup_solver(assemble_A, assemble_b)
        self.linear_solver.solve(self.A, x_star, self.b)
        self.x[:] = x_star[:]
        self.b[:] = self.bold[:]
        self.update()
        return rv, x_star

    def add_exterior(self, **kwargs):
        """
        Set the weak boundary condition.
        This solver requires no modification for Outlet or ConstantPressure
        because n*grad(u) should be zero and thus left out.
        """
        return False


############################ Velocity ################################

class VelocityUpdate_101(VelocityUpdateBase):
    """ 
    Optimized velocity update 
    Just update, no underrelaxation.
    """
    def form(self, u_, u, v, p, q, p_, u_1, u_2, nu, nut_, f, dt, convection_form,
             nuM, **kwargs): 
        self.prm['reassemble_lhs'] = False # No need to reassemble the mass matrix
        self.a = inner(u, v)*dx
        self.aP = inner(v, grad(p))*dx
        self.dt = dt
        self.A = assemble(self.a, tensor = self.A)        
        self.P = assemble(self.aP)        
        [bc.apply(self.A) for bc in self.bcs]
        self.w1 = Vector(self.x)
        self.b = Vector(self.x)
        return False
            
    def solve_Picard_system(self, assemble_A, assemble_b):
        self.prepare()
        self.b[:] = self.A*self.x
        self.b.axpy(-self.dt(0), self.P*(self.solver_namespace['dpx']))
        [bc.apply(self.b) for bc in self.bcs]
        self.w1[:] = self.x[:]
        self.setup_solver(assemble_A, assemble_b)
        self.linear_solver.solve(self.A, self.w1, self.b)
        self.x[:] = self.w1[:]
        self.work[:] = self.x[:] - self.w1[:]            
        self.update()
        return 0., self.work

class Transient_Pressure_101(PressureBase):
    """ Optimized pressure solver """ 
    
    def form(self, u_, u, v, p, q, p_, u_1, u_2, nu, nut_, f, dt, convection_form,
             nuM, **kwargs): 
        # Pressure correction
        self.a = inner(grad(q), dt*grad(p))*dx
        self.aR = inner(q, div(u))*dx
        self.R = assemble(self.aR)
        self.A = assemble(self.a)
        self.A.initialized = True
        self.A1 = self.A.copy()
        self.b = Vector(self.V.dim())
        [bc.apply(self.A) for bc in self.bcs]
        self.prm['reassemble_lhs'] = False
        return False
        
    def solve_Picard_system(self, assemble_A, assemble_b):
        self.prepare()
        self.b[:] = self.A1*self.x
        self.b.axpy(-1., self.R*self.solver_namespace['x_']['u']) # Divergence of u_
        [bc.apply(self.b) for bc in self.bcs]
        #if self.normalize: self.normalize(self.b)
        self.rp = residual(self.A, self.x, self.b)
        self.work[:] = self.x[:]
        self.setup_solver(assemble_A, assemble_b)
        self.linear_solver.solve(self.A, self.x, self.b)
        if self.normalize: self.normalize(self.x)
        self.update()
        return self.rp, self.x - self.work
        
####################### Pressure  #########################################
