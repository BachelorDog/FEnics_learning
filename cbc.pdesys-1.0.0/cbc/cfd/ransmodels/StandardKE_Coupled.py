__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-09-06"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"
"""

    Coupled solver for Standard K-Epsilon turbulence model

"""
from StandardKE import *
from cbc.cfd.tools.Wall import QWall

class StandardKE_Coupled(StandardKE):

    def __init__(self, problem, parameters, model='StandardKE'):
        parameters['model'] = model
        StandardKE.__init__(self, 
                            system_composition=[['k', 'e']],
                            problem=problem, 
                            parameters=parameters)
        
    def create_BCs(self, bcs):
        bcu = StandardKE.create_BCs(self, bcs)
        # Overload default Wall behavior using wallfunctions
        for i, bc in enumerate(bcs):
            if bc.type() == 'Wall':
                bcu['ke'].insert(i+1, QWall['ke'](bc, self.y, self.nu(0)))
                bcu['ke'][i+1].type = bc.type
        return bcu
                
    def define(self):
        """Set up linear algebra schemes."""
        StandardKE.define(self)
        classname = self.prm['time_integration'] + '_ke_' + \
                    str(self.prm['pdesubsystem']['ke'])
        self.pdesubsystems['ke'] = eval(classname)(vars(self), ['k', 'e'], 
                                                   bcs=self.bc['ke'])
                        
class Steady_ke_1(TurbModel):
    
    def form(self, k, e, v_k, v_e, k_, e_, nut_, u_, nu, e_d,
                   P_, T_, sigma_e, Ce1, Ce2, **kwargs):
                       
        Fk = (nu + nut_)*inner(grad(v_k), grad(k))*dx \
             + inner(v_k, dot(grad(k), u_))*dx \
             - P_*v_k*dx + (k_*e*e_d + k*e_*(1. - e_d))*(1./k_)*v_k*dx
             
        Fe = (nu + nut_*(1./sigma_e))*inner(grad(v_e), grad(e))*dx \
             + inner(v_e, dot(grad(e), u_))*dx \
             - (Ce1*P_ - Ce2*e)*(1./T_)*v_e*dx
             
        return Fk + Fe

##+ (k_*e*e_d + k*e_*(1. - e_d))*(1./k_)*v_k*dx

class Steady_ke_2(TurbModel):
    
    def form(self, k, e, v_k, v_e, k_, e_, nut_, u_, Sij_, nu, dt, e_d, 
                   P_, sigma_k, sigma_e, Ce1, Ce2, **kwargs):

        Fk = (nu + nut_)*inner(grad(v_k), grad(k))*dx \
             + inner(v_k, dot(grad(k), u_))*dx \
             - P_*v_k*dx \
             + (k_*e*e_d + k*e_*(1. - e_d))*(1./k_)*v_k*dx
        
        Fe = (nu + nut_*(1./sigma_e))*inner(grad(v_e), grad(e))*dx \
             + inner(v_e, dot(grad(e), u_))*dx \
             - (Ce1*P_*e_ - Ce2*e_*e)*(1./k_)*v_e*dx
        
        return Fk + Fe

class Steady_ke_3(TurbModel):
    """ Pseudo-transient """
    def form(self, k_, e_, k, e, v_k, v_e, nu, nut_, u_, Sij_, dt, e_d, 
                   sigma_k, sigma_e, Ce1, Ce2, **kwargs):
        
        Fk = (1./dt)*inner(k - k_,v_k)*dx \
            + (nu + nut_)*inner(grad(v_k), grad(k))*dx \
            + inner(v_k, dot(grad(k), u_))*dx \
            - 2.*inner(grad(u_), Sij_)*nut_*v_k*dx \
            + (k_*e*e_d + k*e_*(1. - e_d))*(1./k_)*v_k*dx 
        
        Fe = (1./dt)*inner(e - e_, v_e)*dx \
            + (nu + nut_*(1./sigma_e))*inner(grad(v_e), grad(e))*dx \
            + inner(v_e, dot(grad(e), u_))*dx \
            - (Ce1*2.*inner(grad(u_), Sij_)*nut_*e_ - Ce2*e_*e)*(1./k_)*v_e*dx
        
        return Fk + Fe
        
class Steady_ke_4(TurbModel):
    
    def form(self, k_, e_, k, e, v_k, v_e, nu, nut_, u_, Sij_, dt, e_d, 
                   sigma_k, sigma_e, Ce1, Ce2, Pk_, T_, **kwargs):
        F = (nu + nut_)*inner(grad(v_k), grad(k))*dx \
            + inner(v_k, dot(grad(k), u_))*dx \
            - Pk_*k_*v_k*dx \
            + (k_*e*e_d + k*e_*(1. - e_d))*(1./k_)*v_k*dx \
            + (nu + nut_*(1./sigma_e))*inner(grad(v_e), grad(e))*dx \
            + inner(v_e, dot(grad(e), u_))*dx \
            - (Ce1*Pk_*k_ - Ce2*e)*(1/T_)*v_e*dx
        return F
