__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-09-06"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"
"""

    V2F turbulence model
    The two systems (k and epsilon) and (v2 and f) are individually solved coupled

"""
from V2F import *
from cbc.cfd.tools.Wall import V2FWall

class V2F_FullyCoupled(V2F):
    
    def __init__(self, problem, parameters, model='OriginalV2F'):        
        parameters['model'] = model
        parameters['space']['Rij'] = TensorFunctionSpace
        V2F.__init__(self,
                     system_composition=[['k', 'e', 'v2', 'f']],
                     problem=problem,
                     parameters=parameters)
                
    def define(self):
        V2F.define(self)
        self.pdesubsystems['kev2f'] = eval(self.prm['time_integration'] + \
            '_kev2f_' + str(self.prm['pdesubsystem']['kev2f']))(vars(self),
                            self.system_composition[0], bcs=self.bc['kev2f'])
        
    def create_BCs(self, bcs):
        # Set regular boundary conditions
        bcu = V2F.create_BCs(self, bcs)
        # Set wall functions
        for bc in bcs:
            if bc.type() == 'Wall':
                bcu['kev2f'].append(V2FWall(bc, self.y, self.nu(0), 
                                    self.kev2f_))
                bcu['kev2f'][-1].type = bc.type
        return bcu
        
class V2FBase(TurbModel):
    
    def update(self):
        """ Only v2 that is bounded by zero """
        dim = self.solver_namespace['V']['f'].dim()
        x = self.x.array()
        x[:-dim] = minimum(maximum(1e-12, x[:-dim]), 1e6)
        self.x.set_local(x)

# No implicit coupling between k/e and v2/f systems.  

class Steady_kev2f_1(V2FBase):
    def form(self, k, e, v2, f, v_k, v_e, v_v2, v_f, \
                   k_, e_, v2_, f_, nut_, u_, Ce1_, P_, T_, v2ok_, L_, \
                   nu, Ce2, NN, C1, C2, e_d, sigma_e, Cmu, **kwargs):        
        Fke = (nu + nut_)*inner(grad(v_k), grad(k))*dx \
            + inner(v_k, inner(grad(k), u_))*dx \
            - P_*v_k*dx \
            + (k_*e*e_d + k*e_*(1. - e_d))*(1./k_)*v_k*dx \
            + (nu + nut_*(1./sigma_e))*inner(grad(v_e), grad(e))*dx \
            + inner(v_e, inner(grad(e), u_))*dx \
            - (Ce1_*P_ - Ce2*e)*(1/T_)*v_e*dx
            
        Fv2f = (nu + nut_)*inner(grad(v_v2), grad(v2))*dx \
            + inner(v_v2, dot(grad(v2), u_))*dx \
            - k_*f*v_v2*dx + (NN + 1.)*v2ok_*v2/v2_*e_*v_v2*dx \
            + inner(grad(v_f*L_**2), grad(f))*dx \
            + f*v_f*dx  \
            + (1./T_)*((C1 - 1.)*(v2ok_*v2/v2_ - 2./3.) - NN*v2ok_)*v_f*dx \
            - C2*P_/k_*v_f*dx
            
        return Fke + Fv2f

# Introduce some coupling.
class Steady_kev2f_2(V2FBase):
    def form(self, k, e, v2, f, v_k, v_e, v_v2, v_f, \
                   k_, e_, v2_, f_, nut_, u_, Ce1_, P_, T_, v2ok_, L_, \
                   nu, Ce2, NN, C1, C2, e_d, sigma_e, Cmu, **kwargs):        
        Fke = 0.5*(nu + nut_)*inner(grad(v_k), grad(k))*dx \
            + 0.5*(nu + Cmu*v2*T_)*inner(grad(v_k), grad(k_))*dx \
            + inner(v_k, inner(grad(k), u_))*dx \
            - P_*v_k*dx \
            + (k_*e*e_d + k*e_*(1. - e_d))*(1./k_)*v_k*dx \
            + 0.5*(nu + nut_*(1./sigma_e))*inner(grad(v_e), grad(e))*dx \
            + 0.5*(nu + Cmu*v2*T_*(1./sigma_e))*inner(grad(v_e),grad(e_))*dx \
            + inner(v_e, inner(grad(e), u_))*dx \
            - (Ce1_*P_ - Ce2*e)*(1/T_)*v_e*dx
            
        Fv2f = (nu + nut_)*inner(grad(v_v2), grad(v2))*dx \
            + inner(v_v2, dot(grad(v2), u_))*dx \
            - k_*f*v_v2*dx + (NN + 1.)*v2ok_*v2/v2_*e_*v_v2*dx \
            + inner(grad(v_f*L_**2), grad(f))*dx \
            + f*v_f*dx  \
            + (1./T_)*((C1 - 1.)*(v2ok_*v2/v2_ - 2./3.) - NN*v2ok_)*v_f*dx \
            - C2*P_/k_*v_f*dx
            
        return Fke + Fv2f
        
# Psuedo-transient and no implicit coupling between k/e and v2/f systems.                
class Steady_kev2f_3(V2FBase):
    def form(self, k, e, v2, f, v_k, v_e, v_v2, v_f, \
                   k_, e_, v2_, f_, nut_, u_, Ce1_, P_, T_, v2ok_, L_, \
                   nu, dt, Ce2, NN, C1, C2, e_d, sigma_e, **kwargs):        
        Fke = (1./dt)*inner(k - k_,v_k)*dx \
            + (nu + nut_)*inner(grad(v_k), grad(k))*dx \
            + inner(v_k, inner(grad(k), u_))*dx \
            - P_*v_k*dx \
            + (k_*e*e_d + k*e_*(1. - e_d))*(1./k_)*v_k*dx \
            + (1./dt)*inner(e - e_,v_e)*dx \
            + (nu + nut_*(1./sigma_e))*inner(grad(v_e), grad(e))*dx \
            + inner(v_e, inner(grad(e), u_))*dx \
            - (Ce1_*P_ - Ce2*e)*(1/T_)*v_e*dx
            
        Fv2f = (1./dt)*inner(v2 - v2_,v_v2)*dx \
            + (nu + nut_)*inner(grad(v_v2), grad(v2))*dx \
            + inner(v_v2, dot(grad(v2), u_))*dx \
            - k_*f*v_v2*dx + (NN + 1.)*v2ok_*v2/v2_*e_*v_v2*dx \
            + inner(grad(v_f*L_**2), grad(f))*dx \
            + f*v_f*dx  \
            + (1./T_)*((C1 - 1.)*(v2ok_*v2/v2_ - 2./3.) - NN*v2ok_)*v_f*dx \
            - C2*Pk_*v_f*dx
            
        return Fke + Fv2f
