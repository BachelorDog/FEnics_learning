'''
Created on Jul 27, 2016

@author: uvilla
'''
import dolfin as dl
import numpy as np
from helpers_mesh_metric import mesh_metric, hinv_u, h_u, h_dot_u, h_over_u, h_u2
from hippylib import *

class RANS_AlgModel:
    """
    The a RANS algebraic model.
    The state variable is x = [u, p]
    """
    def __init__(self, Vh, nu, ds_ff, u_ff, inletMomentun, inletWidth):
        """
        Contructor
        
        INPUTS:
        - Vh: the mixed finite element space for the state variable
        - nu: the kinetic viscosity
        - ds_ff: the integrator on the farfield boundary (where we impose the swithcing Dir/Neu conditions
        - u_ff: the farfield Dirichlet condition for the normal component of the velocity field u (when backflow is detected)
        - inletMomentun, inletWidth: the standard parameters in the algebraic closure model
        """
        self.Vh = Vh
        self.mesh = Vh.mesh()
        self.nu = nu
        
        self.ds_ff = ds_ff
        self.u_ff = u_ff
        
        self.inletMomentun = inletMomentun
        self.inletWidth = inletWidth
        
        self.metric = mesh_metric(self.mesh)
        self.n = dl.FacetNormal(self.mesh)
        self.tg = dl.perp(self.n)
                        
        self.reg_norm = dl.Constant(1e-1)
        self.Cd = dl.Constant(1e5)
        
        #self.xfun = dl.Expression("x[0]", element = self.Vh.sub(1).ufl_element())
        self.xfun, self.yfun = dl.SpatialCoordinate(self.mesh)
    
    def _u_norm(self,u):
        return dl.sqrt( dl.dot(u,u) + self.reg_norm*self.nu*self.nu)
    
#    def to_uniform_ab(self, a, b, z):
#        """
#        Take z ~ N(0,1) and send it to u ~ U(a,b).
#        """
#        return dl.Constant(b+a)*dl.Constant(0.5) + dl.Constant(b-a)*dl.Constant(0.5)*dl.erf(z/dl.Constant(np.sqrt(2.)))
    
    def _C(self, m):
        return dl.Constant(0.01655963)*dl.exp(m[0])
 #       return self.to_uniform_ab(1e-3, 2e-2, m[0])
        
    def _a(self, m):
        return dl.Constant(4.11619918)*dl.exp(m[1]-dl.Constant(2.)*m[0])
#        return self.to_uniform_ab(0., 2.6, m[1])
        
    def strain(self,u):
        """
        Strain tensor: s = .5( u_{i,j} + u_{j,i})
        """
        return dl.sym(dl.grad(u))
    
    def nu_t(self,m):
        """
        The turbulent viscosity nu_t = C*sqrt(inletMomentum)*(x+a*inletWidth)^{1/2}
        where C = exp(m[0]) and a = exp( m[1] )
        """
        #C = dl.exp(m[0])
        #a = dl.exp(m[1])
        C = self._C(m)
        a = self._a(m)
        return C*dl.sqrt(self.inletMomentun)*dl.sqrt(self.xfun + a*self.inletWidth)
                    
    def sigma_n(self,nu,u):
        """
        The boundary stress tensor
        """
        return dl.dot( dl.Constant(2.)*nu*self.strain(u), self.n )
        
    
    def tau(self,nu, u, metric):
        """
        Stabilization parameter
        """
        h2 = h_u2(self.metric, u, self.reg_norm*self.nu*self.nu)
        Pe =  dl.Constant(.5)*h_dot_u(metric, u, self.reg_norm*self.nu*self.nu)/nu
                    
        num = dl.Constant(1.) + dl.exp(dl.Constant(-2.)*Pe)
        den = dl.Constant(1.) - dl.exp(dl.Constant(-2.)*Pe)
        
        # [0.1 0.01]* [a1] = [ coth(.1) - 1./(.1) ]
        # [1.  0.2 ]  [a2]   [ -csch(.1)^2 + 1./(.1)^2]

        a1 = dl.Constant(0.333554921691650)
        a2 = dl.Constant(-0.004435991517475)
            
        tau_1 = (num/den - dl.Constant(1.)/Pe)*h_over_u(metric, u, self.reg_norm*self.nu*self.nu)
        tau_2 = (a1 + a2*Pe)*dl.Constant(.5)*h2/nu       
                
        return dl.conditional(dl.ge(Pe, .1), tau_1, tau_2)

    
    def all_tau(self, x, m):
        """
        All the 4 stabilization parameters
        """
        u,p = dl.split(x) 
        h2 = h_u2(self.metric, u, self.reg_norm*self.nu*self.nu)
        tau = [ self.tau(self.nu + self.nu_t(m), u, self.metric),
                h2*self._u_norm(u)
               ]
        return tau
    
    def stab(self,x, x_test, m, true_derivative):
        """
        The G-LS stabilization
        """
        r_s = self.strong_residual(x, m)
        
        r_s_prime =[None, None]
        
        if true_derivative:
            for i in range(2):
                r_s_prime[i] = dl.derivative(r_s[i], x, x_test)
        
            tau = self.all_tau(x,m)
        else:
            xl = vector2Function(x.vector(), self.Vh)
            r_sl = self.strong_residual(xl, m)
            for i in range(2):
                r_s_prime[i] = dl.derivative(r_sl[i], xl, x_test)
        
            tau = self.all_tau(xl,m)
        
        res_stab = ( tau[0]*dl.inner(r_s[0], r_s_prime[0]) + \
                     tau[1]*dl.inner(r_s[1], r_s_prime[1]) )*dl.dx
                                         
        return res_stab
                
    def weak_residual(self, x, x_test, m):
        """
        The weak residual
        """
        u,p = dl.split(x)
        u_test, p_test = dl.split(x_test)
        
        hbinv = hinv_u(self.metric, self.tg)
        
        res_u =  dl.Constant(2.)*(self.nu+self.nu_t(m))*dl.inner( self.strain(u), self.strain(u_test))*dl.dx \
               + dl.inner(dl.grad(u)*u, u_test)*dl.dx \
               - p*dl.div(u_test)*dl.dx \
               + self.Cd*(self.nu+self.nu_t(m))*hbinv*dl.dot(u - self.u_ff, self.tg)*dl.dot(u_test, self.tg)*self.ds_ff \
               - dl.dot( self.sigma_n(self.nu+self.nu_t(m), u), self.tg) * dl.dot(u_test, self.tg)*self.ds_ff \
               - dl.dot( self.sigma_n(self.nu+self.nu_t(m), u_test), self.tg ) * dl.dot(u - self.u_ff, self.tg)*self.ds_ff
  
               
        res_p =  dl.div(u)*p_test*dl.dx
                          
        return res_u + res_p
    
    def strong_residual(self, x, m):
        """
        The strong residual
        """
        u,p = dl.split(x)
         
        res_u = -dl.div( dl.Constant(2.)*(self.nu+self.nu_t(m))*self.strain(u) ) + dl.grad(u)*u + dl.grad(p)
        res_p = dl.div( u )
                
        return [res_u, res_p]
    
                    
    def residual(self, x, x_test, m, true_derivative=True):
        """
        Returns the weak form of residual.
        
        INPUTS:
        - x: the state at which evaluate the Jacobian
        - x_test: the test function
        - m: the uncertain parameter
        """
                               
        return self.weak_residual(x, x_test, m) + self.stab(x, x_test, m, true_derivative)

        
    def Jacobian(self, x, x_test, x_trial, m, true_derivative=True):
        """
        Returns the weak form of the Jacobian matrix.
        
        INPUTS:
        - x: the state at which evaluate the Jacobian
        - x_test: the test function
        - x_trial: the trial function
        - m: the uncertain parameter
        """
        r = self.residual(x, x_test, m, true_derivative)
        return dl.derivative(r,x,x_trial)
    
    def mass(self,x_test,x_trial, coeffs=None):
        """
        Assemble an auxiliary mass matrix
        """
        if coeffs is None:
            coeffs = [dl.Constant(1.), dl.Constant(0.)]
        u_test, p_test = dl.split(x_test)
        u_trial, p_trial = dl.split(x_trial)
        return coeffs[0]*dl.inner(u_trial,u_test)*dl.dx + coeffs[1]*p_trial*p_test*dl.dx
               
    def stiffness(self,x_test,x_trial, coeffs=None):
        """
        Assemble an auxiliary stiffness matrix
        """
        if coeffs is None:
            coeffs = [dl.Constant(1.), dl.Constant(0.)]
        u_test, p_test = dl.split(x_test)
        u_trial, p_trial = dl.split(x_trial)
        
        return coeffs[0]*dl.inner(self.strain(u_trial), self.strain(u_test))*dl.dx \
               + coeffs[1]*p_trial*p_test*dl.dx