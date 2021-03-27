__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-09-14"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"
"""

    Segregated LowReynolds turbulence models

"""
from LowReynolds import *

class LowReynolds_Segregated(LowReynolds):

    def __init__(self, problem, parameters, model='LaunderSharma'):
        parameters['model'] = model
        LowReynolds.__init__(self, 
                             system_composition=[['k'],['e']],
                             problem=problem,
                             parameters=parameters)
        
    def define(self):
        """ Set up linear algebra schemes and their boundary conditions """
        LowReynolds.define(self)
        self.schemes['k'] = eval(self.prm['time_integration'] + '_k_' + 
               str(self.prm['scheme']['k']))(self, self.system_composition[0])        
        self.schemes['e'] = eval(self.prm['time_integration'] + '_e_' + 
               str(self.prm['scheme']['e']))(self, self.system_composition[1])
        
class Steady_k_1(TurbModel):
        
    def form(self, k, v_k, k_, e_, nut_, D_, u_, Sij_, nu,  **kwargs):
        F = (nu + nut_)*inner(grad(v_k), grad(k))*dx \
            + inner(v_k, dot(grad(k), u_))*dx \
            - 2.*inner(grad(u_), Sij_)*nut_*v_k*dx \
            + k*e_*(1./k_)*v_k*dx + v_k*D_*dx
        return F

class Steady_k_2(TurbModel):
    """Pseudo-Transient."""    
    def form(self, k, v_k, k_, e_, nut_, D_, u_, Sij_, nu, dt,  **kwargs):
        F = (1./dt)*inner(k - k_, v_k)*dx \
            + (nu + nut_)*inner(grad(v_k), grad(k))*dx \
            + inner(v_k, dot(grad(k), u_))*dx \
            - 2.*inner(grad(u_), Sij_)*nut_*v_k*dx \
            + k*e_*(1./k_)*v_k*dx + v_k*D_*dx
        return F

class Steady_e_1(TurbModel):
    
    def form(self, e, v_e, e_, k_, nut_, E0_, f2_, u_, Sij_, sigma_e, Ce1, Ce2, 
             nu, **kwargs):
        F = (nu + nut_*(1./sigma_e))*inner(grad(v_e), grad(e))*dx \
            + inner(v_e, dot(grad(e), u_))*dx \
            - (Ce1*2.*inner(grad(u_), Sij_)*nut_*e_ \
            - f2_*Ce2*e_*e)*(1./k_)*v_e*dx \
            - E0_*v_e*dx
        return F

class Steady_e_2(TurbModel):
    """Pseudo-Transient."""
    def form(self, e, v_e, e_, k_, nut_, E0_, f2_, u_, Sij_, sigma_e, Ce1, Ce2, 
             dt, nu, **kwargs):
        F = (1./dt)*inner(e-e_,v_e)*dx \
            + (nu + nut_*(1./sigma_e))*inner(grad(v_e), grad(e))*dx \
            + inner(v_e, dot(grad(e), u_))*dx \
            - (Ce1*2.*inner(grad(u_), Sij_)*nut_*e_ \
            - f2_*Ce2*e_*e)*(1./k_)*v_e*dx \
            - E0_*v_e*dx
        return F
