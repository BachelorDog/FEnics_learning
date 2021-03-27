import dolfin as dl
import numpy as np
from helpers_mesh_metric import mesh_metric, hinv_u, h_u, h_dot_u, h_over_u, h_u2
from hippylib import *

class RANSModel_inadeguate_Cmu:
    """
    The basic RANS model with k-e closure.
    This model is for the solution of the fwd problem WITHOUT uncertain or control parameters.
    The state variable is x = [u, p, k, e].
    The model is written for transient continuation.
    The discretization scheme in time is a semi-implicit Euler scheme, where
    the turbulent production is treated explicitely.
    """
    def __init__(self, Vh, nu, ds_ff,
                 u_ff, k_ff, e_ff,
                 C_mu = dl.Constant(0.09),
                 sigma_k = dl.Constant(1.00),
                 sigma_e = dl.Constant(1.30),
                 C_e1 = dl.Constant(1.44),
                 C_e2 = dl.Constant(1.92)
                 ):
        """
        Contructor
        
        INPUTS:
        - Vh: the mixed finite element space for the state variable
        - nu: the kinetic viscosity
        - ds_ff: the integrator on the farfield boundary (where we impose the swithcing Dir/Neu conditions
        - u_ff: the farfield Dirichlet condition for the normal component of the velocity field u (when backflow is detected)
        - k_ff: the farfield Dirichlet condition for the normal component of the turbulent kinetic energy field k (when backflow is detected)
        - e_ff: the farfield Dirichlet condition for the normal component of the turbulent energy dissipation field e (when backflow is detected)
        - C_mu, sigma_k, sigma_e, C_e1, C_e1: the standard parameters in the KE closure model
        """
        self.Vh = Vh
        self.mesh = Vh.mesh()
        self.nu = nu
        
        self.ds_ff = ds_ff
        self.u_ff = u_ff
        self.k_ff = k_ff
        self.e_ff = e_ff
        
        self.C_mu = C_mu
        self.sigma_k =sigma_k
        self.sigma_e =sigma_e
        self.C_e1 = C_e1
        self.C_e2 = C_e2
        
        self.metric = mesh_metric(self.mesh)
        self.n = dl.FacetNormal(self.mesh)
        self.tg = dl.perp(self.n)
        
        self.reg_k = dl.Constant(1e-8)
        self.reg_e = dl.Constant(1e-8)
                
        self.reg_norm = dl.Constant(1e-1)
        self.beta_chi_inflow = dl.Constant(1.)/dl.sqrt( self.nu )
        self.shift_chi_inflow = dl.Constant(0.)
        self.Cd = dl.Constant(1e5)
        
    def _k_plus(self,k):
        return dl.Constant(0.5)*( k + dl.sqrt(k*k+self.reg_k) )
    
    def _e_plus(self,e):
        return dl.Constant(0.5)*( e + dl.sqrt(e*e+self.reg_e) )
    
    def _chi_inflow(self,u):
        return dl.Constant(.5) - dl.Constant(.5)*dl.tanh(self.beta_chi_inflow*dl.dot(u, self.n) - self.shift_chi_inflow)

    def _u_norm(self,u):
        return dl.sqrt( dl.dot(u,u) + self.reg_norm*self.nu*self.nu)
    
    def _theta(self,k,e):
        kp = self._k_plus(k)
        return e/kp
        
    def strain(self,u):
        """
        Strain tensor: s = .5( u_{i,j} + u_{j,i})
        """
        return dl.sym(dl.grad(u))
    
    def nu_t(self,k,e,m):
        """
        The turbulent viscosity nu_t = C_my * k^2/e
        """
        ep = self._e_plus(e)
        return dl.exp(m)*self.C_mu*k*k/ep
            
    def production(self,u):
        """
        The turbulent production term P = .5*( strain(u):strain(u) )
        """
        return dl.Constant(2.)*dl.inner(self.strain(u), self.strain(u))
        
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
        u,p,k,e = dl.split(x) 
        h2 = h_u2(self.metric, u, self.reg_norm*self.nu*self.nu)
        tau = [ self.tau(self.nu + self.nu_t(k,e,m), u, self.metric),
                h2*self._u_norm(u),
                self.tau( self.nu + self.nu_t(k,e,m)/self.sigma_k, u, self.metric),
               self.tau( self.nu + self.nu_t(k,e,m)/self.sigma_e, u, self.metric)
               ]
        return tau
    
    def stab(self,x, x_test, m, x0, xl, true_derivative=False):
        """
        The G-LS stabilization
        """
        r_s = self.strong_residual(x, m, xl)
        
        r_s_prime =[None, None, None, None]
        
        #if true_derivative:
        #    for i in range(4):
        #        r_s_prime[i] = dl.derivative(r_s[i], x, x_test) + dl.derivative(r_s[i], xl, x_test)
        #else:
        #    for i in range(4):
        #        r_s_prime[i] = dl.derivative(r_s[i], x, x_test)
        
        for i in range(4):
            r_s_prime[i] = dl.derivative(r_s[i], x, x_test)
        
        tau = self.all_tau(xl,m)
        
        res_stab = ( tau[0]*dl.inner(r_s[0], r_s_prime[0]) + \
                     tau[1]*dl.inner(r_s[1], r_s_prime[1]) + \
                     tau[2]*dl.inner(r_s[2], r_s_prime[2]) + \
                     tau[3]*dl.inner(r_s[3], r_s_prime[3]) )*dl.dx
                                         
        return res_stab
                
    def weak_residual(self, x, x_test, sigma, m, x0, xl):
        """
        The weak residual
        """
        u,p,k,e = dl.split(x)
        u0,p0,k0,e0 = dl.split(x0)
        ul,pl,kl,el = dl.split(xl)

        u_test, p_test, k_test, e_test = dl.split(x_test)
        
        hbinv = hinv_u(self.metric, self.tg)
        
        res_u =  dl.Constant(2.)*(self.nu+self.nu_t(k,e,m))*dl.inner( self.strain(u), self.strain(u_test))*dl.dx \
               + dl.inner(dl.grad(u)*u, u_test)*dl.dx \
               - p*dl.div(u_test)*dl.dx \
               + self.Cd*(self.nu+self.nu_t(k,e,m))*hbinv*dl.dot(u - self.u_ff, self.tg)*dl.dot(u_test, self.tg)*self.ds_ff \
               - dl.dot( self.sigma_n(self.nu+self.nu_t(k,e,m), u), self.tg) * dl.dot(u_test, self.tg)*self.ds_ff \
               - dl.dot( self.sigma_n(self.nu+self.nu_t(k,e,m), u_test), self.tg ) * dl.dot(u - self.u_ff, self.tg)*self.ds_ff
               #- self._chi_inflow(ul)*dl.Constant(.5)*dl.inner( dl.dot(u,u)*self.n + dl.dot(self.n,u)*u, u_test )*self.ds_ff \
  
               
        res_p =  dl.div(u)*p_test*dl.dx
               
        
        res_k = dl.dot(u, dl.grad(k))*k_test*dl.dx \
                + (self.nu + self.nu_t(k,e,m)/self.sigma_k)*dl.inner( dl.grad(k), dl.grad(k_test))*dl.dx \
                + e*k_test*dl.dx \
                - self.nu_t(kl,el,m)*self.production(ul)*k_test*dl.dx \
                - self._chi_inflow(ul)*(self.nu + self.nu_t(k,e,m)/self.sigma_k)*dl.dot(dl.grad(k), self.n)*k_test*self.ds_ff \
                - self._chi_inflow(ul)*(self.nu + self.nu_t(k,e,m)/self.sigma_k)*dl.dot(dl.grad(k_test), self.n)*(k-self.k_ff)*self.ds_ff \
                + self._chi_inflow(ul)*self.Cd*(self.nu + self.nu_t(k,e,m)/self.sigma_k)*hbinv*(k-self.k_ff)*k_test*self.ds_ff
                
        res_e = dl.dot(u, dl.grad(e))*e_test*dl.dx \
                + (self.nu + self.nu_t(k,e,m)/self.sigma_e)*dl.inner( dl.grad(e), dl.grad(e_test))*dl.dx \
                + self.C_e2*e*self._theta(k,e)*e_test*dl.dx \
                - self.C_e1*dl.exp(m)*self.C_mu*kl*self.production(ul)*e_test*dl.dx \
                - self._chi_inflow(ul)*(self.nu + self.nu_t(k,e,m)/self.sigma_e)*dl.dot(dl.grad(e), self.n)*k_test*self.ds_ff \
                - self._chi_inflow(ul)*(self.nu + self.nu_t(k,e,m)/self.sigma_e)*dl.dot(dl.grad(e_test), self.n)*(e-self.e_ff)*self.ds_ff \
                + self._chi_inflow(ul)*self.Cd*(self.nu + self.nu_t(k,e,m)/self.sigma_e)*hbinv*(e-self.e_ff)*e_test*self.ds_ff
        
        time_derivative = sigma*(
                            dl.inner( u - u0, u_test) + dl.inner(k - k0, k_test) + dl.inner(e - e0, e_test) )*dl.dx
                
        return time_derivative + res_u + res_p + res_k + res_e
    
    def strong_residual(self, x, m, xl):
        """
        The strong residual
        """
        u,p,k,e = dl.split(x)
        ul, pl, kl, el = dl.split(xl)
         
        res_u = -dl.div( dl.Constant(2.)*(self.nu+self.nu_t(kl,el,m))*self.strain(u) ) + dl.grad(u)*ul + dl.grad(p)
        res_p = dl.div( u )
        res_k = + dl.dot(ul, dl.grad(k)) \
                - dl.div( (self.nu + self.nu_t(kl,el,m)/self.sigma_k)* dl.grad(k) ) \
                + el \
                - self.nu_t(kl,el,m)*self.production(ul)
                
        res_e = + dl.dot(ul, dl.grad(e)) \
                - dl.div( (self.nu + self.nu_t(kl,el,m)/self.sigma_e)* dl.grad(e) ) \
                + self.C_e2*e*self._theta(kl,e) \
                - self.C_e1*dl.exp(m)*self.C_mu*kl*self.production(ul)
                
        return [res_u, res_p, res_k, res_e]
    
                    
    def residual(self, x, x_test, sigma, m, x0, xl, true_derivative=False):
        """
        Returns the weak form of residual.
        
        INPUTS:
        - x: the state at which evaluate the Jacobian
        - x_test: the test function
        - sigma: the inverse of the time-step
        - m: the uncertain parameter
        - x0: the state at the previous time step
        - xl: the auxiliary state to compute the Picard Jacobian (if None then xl is a copy of x)
        - true_derivative: if true compute the true derivative of the strong residual in the G-LS
        """
        if x0 is None:
            x0 = vector2Function(x.vector(), self.Vh)
        if xl is None:
            xl = vector2Function(x.vector(), self.Vh)                          
        return self.weak_residual(x, x_test, sigma, m, x0, xl) + self.stab(x, x_test, m, x0, xl, true_derivative)
        
    def Jacobian(self, x, x_test, x_trial, sigma, m, x0, xl, true_derivative = False):
        """
        Returns the weak form of the Jacobian matrix.
        
        INPUTS:
        - x: the state at which evaluate the Jacobian
        - x_test: the test function
        - x_trial: the trial function
        - sigma: the inverse of the time-step
        - m: the uncertain parameter
        - x0: the state at the previous time step
        - xl: the auxiliary state to compute the Picard Jacobian (if None then xl is a copy of x)
        - true_derivative: if true compute the true Jacobian, if false compute the Picard Jacobian
        """
        if x0 is None:
            x0 = vector2Function(x.vector(), self.Vh)
        if xl is None:
            xl = vector2Function(x.vector(), self.Vh)  
        r = self.residual(x, x_test, sigma, m, x0, xl, true_derivative)
        if true_derivative:
            return dl.derivative(r,x,x_trial) + dl.derivative(r, xl, x_trial)
        else:
            return dl.derivative(r,x,x_trial)
    
    def mass(self,x_test,x_trial, coeffs=None):
        """
        Assemble an auxiliary mass matrix
        """
        if coeffs is None:
            coeffs = [dl.Constant(1.), dl.Constant(0.), dl.Constant(1.), dl.Constant(1.)]
        u_test, p_test, k_test, e_test = dl.split(x_test)
        u_trial, p_trial, k_trial, e_trial = dl.split(x_trial)
        return coeffs[0]*dl.inner(u_trial,u_test)*dl.dx + coeffs[1]*p_trial*p_test*dl.dx \
               + coeffs[2]*k_trial*k_test*dl.dx + coeffs[3]*e_test*e_trial*dl.dx
               
    def stiffness(self,x_test,x_trial, coeffs=None):
        """
        Assemble an auxiliary stiffness matrix
        """
        if coeffs is None:
            coeffs = [dl.Constant(1.), dl.Constant(0.), dl.Constant(1.), dl.Constant(1.)]
        u_test, p_test, k_test, e_test = dl.split(x_test)
        u_trial, p_trial, k_trial, e_trial = dl.split(x_trial)
        return coeffs[0]*dl.inner(self.strain(u_trial), self.strain(u_test))*dl.dx \
               + coeffs[1]*p_trial*p_test*dl.dx \
               + coeffs[2]*dl.inner(dl.grad(k_trial), dl.grad(k_test))*dl.dx \
               + coeffs[3]*dl.inner(dl.grad(e_trial), dl.grad(e_test))*dl.dx