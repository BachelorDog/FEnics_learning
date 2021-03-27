__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-09-06"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"
"""

    Spallart Allmaras turbulence model

"""
from TurbSolver import *
from cbc.cfd.tools.Eikonal import Eikonal

class SpalartAllmaras(TurbSolver):
    """Spallart Allmaras turbulence model."""
    def __init__(self, problem, parameters, model='SpalartAllmaras'):
        parameters['model'] = model
        self.classical = True
        TurbSolver.__init__(self, 
                            system_composition=[['nu_tilde']],
                            problem=problem, 
                            parameters=parameters)
                        
    def define(self):
        """Set up linear algebra schemes and their boundary conditions."""
        DQ, DQ_NoBC = DerivedQuantity, DerivedQuantity_NoBC
        V, NS = self.V['dq'], self.problem.NS_solver
        NS.pdesubsystems['derived quantities'] = [
          DQ_NoBC(vars(NS), 'Omega', NS.S, "sqrt(2.*inner(omega(u_), omega(u_)))")]
        self.Omega_ = NS.Omega_
        # Constant
        self.Cw1 = self.Cb1/self.kappa**2 + (1. + self.Cb2)/self.sigma
        ns = vars(self)
        self.pdesubsystems['derived quantities'] = [
              DQ(ns, 'chi', V, "nu_tilde_/nu"),
              DQ(ns, 'fv1', V, "chi_**3/(chi_**3 + Cv1**3)"),
              DQ(ns, 'nut', V, "nu_tilde_*fv1_")]
        self.pdesubsystems['derived quantities'] += {
            True: lambda: [
                DQ(ns, 'fv2', V, "1. - chi_/(1. + chi_*fv1_)", bounded=False),
                DQ_NoBC(ns, 'St', V, "Omega_ + nu_tilde_/(kappa*y)**2*fv2_")],
            False: lambda: [
                DQ(ns, 'fv2', V, "1./(1. + chi_/Cv2)**3", wall_value=1.), 
                DQ(ns, 'fv3', V, "(1. + chi_*fv1_)*(1 - fv2_)/chi_", wall_value=1.),
                DQ_NoBC(ns, 'St', V, "fv3_*Omega_ + nu_tilde_/(kappa*y)**2*fv2_")]
            }[self.classical]()
                
        self.pdesubsystems['derived quantities'] += [
            DQ(ns, 'r', V, "nu_tilde_/(Omega_*kappa**2*y**2 + nu_tilde_*fv2_)",
               wall_value=1.),
            DQ(ns, 'g', V, "r_ + Cw2*(r_**6 - r_)", wall_value=1-self.Cw2(0)),
            DQ(ns, 'fw', V, "g_*((1. + Cw3**6)/(g_**6 + Cw3**6))**(1./6.)", 
               wall_value=1.)]
        
        classname = self.prm['time_integration'] + '_nu_tilde_' + \
                    str(self.prm['pdesubsystem']['nu_tilde'])
        self.pdesubsystems['nu_tilde'] = eval(classname)(vars(self), ['nu_tilde'],
                                                      bcs=self.bc['nu_tilde'])
        
        TurbSolver.define(self)
        
    def model_parameters(self):
        for dq in ['nut', 'r', 'g', 'fw']:
            self.prm['apply'][dq] = self.prm['apply'].get(dq, 'project')
            
        self.model_prm = dict(
            sigma = Constant(2./3.),
            Cv1 = Constant(7.1),
            Cb1 = Constant(0.1355),
            Cb2 = Constant(0.622),
            kappa = Constant(0.4187),
            Cw2 = Constant(0.3),
            Cw3 = Constant(2.),
            Ct3 = Constant(1.2),
            Cv2 = Constant(5.0),
            )
        self.__dict__.update(self.model_prm)
        
    def create_BCs(self, bcs):
        # Compute distance to nearest wall
        self.distance = Eikonal(self.mesh, self.boundaries)
        self.y = self.distance.y_
        return TurbSolver.create_BCs(self, bcs)
        
# Model

class Steady_nu_tilde_1(TurbModel):
    
    def form(self, nu_tilde, v_nu_tilde, nu_tilde_, fw_, St_, u_, Omega_, nu, 
             y, sigma, Cw1, Cb1, Cb2, **kwargs):
        F = (1./sigma)*(nu + nu_tilde_)*inner(grad(nu_tilde), \
                                              grad(v_nu_tilde))*dx \
            - (Cb2/sigma)*inner(v_nu_tilde, dot(grad(nu_tilde_), \
                                                grad(nu_tilde_)))*dx \
            + inner(v_nu_tilde, dot(grad(nu_tilde), u_))*dx \
            + inner(v_nu_tilde, Cw1*fw_*nu_tilde_/y**2*nu_tilde)*dx \
            - inner(v_nu_tilde, Cb1*St_*nu_tilde_)*dx # Production explicit
        return F

class Steady_nu_tilde_2(TurbModel):
    
    def form(self, nu_tilde, v_nu_tilde, nu_tilde_, fw_, St_, u_, Omega_, nu, 
             dt, y, sigma, Cw1, Cb1, Cb2, **kwargs):
        F = (1./dt)*inner(v_nu_tilde, nu_tilde - nu_tilde_)*dx \
            + (1./sigma)*(nu + nu_tilde_)*inner(grad(nu_tilde), \
                                                grad(v_nu_tilde))*dx \
            - (Cb2/sigma)*inner(v_nu_tilde, dot(grad(nu_tilde_), \
                                                grad(nu_tilde_)))*dx \
            + inner(v_nu_tilde, dot(grad(nu_tilde), u_))*dx \
            + inner(v_nu_tilde, Cw1*fw_*nu_tilde_/y**2*nu_tilde)*dx \
            - inner(v_nu_tilde, Cb1*St_*nu_tilde_)*dx # Production explicit
        return F
