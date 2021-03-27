__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-09-06"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"
"""

    Coupled LowReynolds turbulence models

"""
from LowReynolds import *

class LowReynolds_Coupled(LowReynolds):

    def __init__(self, problem, parameters, model='LaunderSharma'):
        parameters['model'] = model
        LowReynolds.__init__(self, 
                             system_composition=[['k','e']],
                             problem=problem,
                             parameters=parameters)
                        
    def define(self):
        """ Set up linear algebra schemes and their boundary conditions """
        LowReynolds.define(self)
        classname = '{}_ke_{}'.format(self.prm['time_integration'], 
                                      self.prm['pdesubsystem']['ke'])
        self.pdesubsystems['ke'] = eval(classname)(vars(self), ['k', 'e'],
                                                   bcs=self.bc['ke'])

class Steady_ke_1(TurbModel):
    
    def form(self, k, e, v_k, v_e, k_, e_, nut_, u_, Sij_, E0_, f2_, D_, 
                   nu, e_d, sigma_e, Ce1, Ce2, **kwargs):
        Fk = (nu + nut_)*inner(grad(v_k), grad(k))*dx + \
             inner(v_k, dot(grad(k), u_))*dx - \
             2.*inner(grad(u_), Sij_)*nut_*v_k*dx + \
             (k_*e*e_d + k*e_*(1. - e_d))*(1./k_)*v_k*dx + v_k*D_*dx
            
        Fe = (nu + nut_*(1./sigma_e))*inner(grad(v_e), grad(e))*dx + \
             inner(v_e, dot(grad(e), u_))*dx - \
             (Ce1*2.*inner(grad(u_), Sij_)*nut_*e_ - \
             f2_*Ce2*e_*e)*(1./k_)*v_e*dx - \
             E0_*v_e*dx            
            
        return Fk + Fe
        
class Steady_ke_2(TurbModel):
    """ Pseudo-transient """
    def form(self, k, e, v_k, v_e, k_, e_, nut_, u_, Sij_, E0_, f2_, D_,
                   nu, dt, e_d, sigma_e, Ce1, Ce2, **kwargs):
        Fk = (1./dt)*inner(k - k_, v_k)*dx + \
             (nu + nut_)*inner(grad(v_k), grad(k))*dx + \
             inner(v_k, dot(grad(k), u_))*dx - \
             2.*inner(grad(u_), Sij_)*nut_*v_k*dx + \
             (k_*e*e_d + k*e_*(1. - e_d))*(1./k_)*v_k*dx + v_k*D_*dx
             
        Fe = (nu + nut_*(1./sigma_e))*inner(grad(v_e), grad(e))*dx + \
             inner(v_e, dot(grad(e), u_))*dx - \
             (Ce1*2.*inner(grad(u_), Sij_)*nut_*e_ - \
             f2_*Ce2*e_*e)*(1./k_)*v_e*dx - \
             E0_*v_e*dx
             
        return Fk + Fe
        