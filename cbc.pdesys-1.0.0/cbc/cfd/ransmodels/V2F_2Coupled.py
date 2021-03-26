__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-09-06"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"
"""

    V2F turbulence model
    The two systems (k and epsilon) and (v2 and f) are individually solved coupled

"""
from V2F import *
from cbc.cfd.tools.Wall import QWall

class V2F_2Coupled(V2F):
    
    def __init__(self, problem, parameters, model='OriginalV2F'):
        # A segregated system of two coupled systems:
        parameters['model'] = model
        parameters['space']['Rij'] = TensorFunctionSpace
        V2F.__init__(self,
                     system_composition=[['k', 'e'], ['v2', 'f']],
                     problem=problem,
                     parameters=parameters)
                
    def define(self):
        V2F.define(self)
        self.pdesubsystems['ke']  = eval(self.prm['time_integration'] + '_ke_' + 
                                    str(self.prm['pdesubsystem']['ke']))(vars(self),
                                       ['k', 'e'], bcs=self.bc['ke'])
        self.pdesubsystems['v2f'] = eval(self.prm['time_integration'] + '_v2f_' + 
                                    str(self.prm['pdesubsystem']['v2f']))(vars(self), 
                                       ['v2', 'f'], bcs=self.bc['v2f'])
        
    def create_BCs(self, bcs):
        # Set regular boundary conditions
        bcu = V2F.create_BCs(self, bcs)
        # Set wall functions
        for bc in bcs:
            if bc.type() == 'Wall':
                bcu['ke'].append(QWall['ke'](bc, self.y, self.nu(0))) 
                bcu['ke'][-1].type = bc.type
                if self.prm['model'] == 'OriginalV2F':
                    bcu['v2f'].append(QWall['v2f'](bc, self.y, self.nu(0), 
                                                   self.ke_))
                    bcu['v2f'][-1].type = bc.type
        return bcu
        
    def solve_inner(self, max_iter=1, max_err=1e-7, logging=True):
        for name in self.system_names:
            err, j = solve_nonlinear([self.pdesubsystems[name]],
                                     max_iter=max_iter, max_err=max_err,
                                     logging=max_iter>1)
            self.solve_derived_quantities()
        self.total_number_iters += j
        return err
                
class V2FBase(TurbModel):
    
    def update(self):
        """ Only v2 that is bounded by zero """
        if self.solver_namespace['prm']['model'] == 'OriginalV2F':
            dim = self.x.size(0)/2
            self.x[:dim].set_local(minimum(maximum(1e-12, self.x.array()[:dim]), 1e8))
        else:
            self.x[:].set_local(minimum(maximum(1e-12, self.x.array()[:]), 1e8))
    
class KEBase(TurbModel):    
    def update(self):
        """This makes k=1e-10 on walls and v2ok is then 0.01 if used with compute_dofs."""
        bound(self.x, minf=1e-10)
        
class Steady_ke_1(KEBase):
    def form(self, k, e, v_k, v_e, k_, e_, nu, nut_, u_, Ce1_, P_, T_, Ce2, 
                   e_d, sigma_e, **kwargs):        
        Fk = (nu + nut_)*inner(grad(v_k), grad(k))*dx \
            + inner(v_k, inner(u_, grad(k)))*dx \
            - P_*v_k*dx + (k_*e*e_d + k*e_*(1. - e_d))*(1./k_)*v_k*dx
            
        Fe = (nu + nut_*(1./sigma_e))*inner(grad(v_e), grad(e))*dx \
            + inner(v_e, inner(u_, grad(e)))*dx \
            - (Ce1_*P_ - Ce2*e)*(1/T_)*v_e*dx
            
        return Fk + Fe
        
class Steady_ke_2(KEBase):
    """Pseudo-transient."""
    def form(self, k, e, v_k, v_e, k_, e_, nu, nut_, u_, Ce1_, P_, T_, Sij_, 
                   dt, Ce2, e_d, sigma_e, **kwargs):
        Fk = (1./dt)*inner(k - k_, v_k)*dx \
            + (nu + nut_)*inner(grad(v_k), grad(k))*dx \
            + inner(v_k, dot(u_, grad(k)))*dx \
            - P_*v_k*dx + (k_*e*e_d + k*e_*(1. - e_d))*(1./k_)*v_k*dx
            
        Fe = (1./dt)*inner(k - k_, v_k)*dx \
            + (nu + nut_*(1./sigma_e))*inner(grad(v_e), grad(e))*dx \
            + inner(v_e, dot(u_, grad(e)))*dx \
            - (Ce1_*2.*inner(grad(u_), Sij_)*nut_ - Ce2*e)*(1/T_)*v_e*dx
            
        return Fk + Fe
        
class Steady_v2f_1(V2FBase):
    
    def form(self, v2, f, v_v2, v_f, v2_, f_, k_, e_, nu, nut_, u_, v2ok_, P_, 
                   T_, L_, NN, C1, C2, **kwargs):
        Fv2 = (nu + nut_)*inner(grad(v_v2), grad(v2))*dx \
            + inner(v_v2, dot(u_, grad(v2)))*dx \
            - k_*f*v_v2*dx + (NN + 1.)*v2ok_*v2/v2_*e_*v_v2*dx
            
        Ff =  inner(grad(v_f*L_**2), grad(f))*dx \
            + f*v_f*dx + (1./T_)*((C1 - 1.)*(v2ok_*v2/v2_ - 2./3.) \
            - NN*v2ok_)*v_f*dx - C2*P_/k_*v_f*dx
            
        return Fv2 + Ff

class Steady_v2f_2(V2FBase):
    """Pseudo-transient."""
    def form(self, v2, f, v_v2, v_f, v2_, f_, k_, e_, nu, dt, nut_, u_, v2ok_, 
                   P_, T_, L_, NN, C1, C2, **kwargs):
        Fv2 = (1./dt)*inner(v2 - v2_, v_v2)*dx \
            + (nu + nut_)*inner(grad(v_v2), grad(v2))*dx \
            + inner(v_v2, dot(grad(v2), u_))*dx \
            - k_*f*v_v2*dx + (NN + 1.)*v2ok_*v2/v2_*e_*v_v2*dx \
        
        Ff = inner(grad(v_f*L_**2), grad(f))*dx \
            + f*v_f*dx + (1./T_)*((C1 - 1.)*(v2ok_*v2/v2_ - 2./3.) \
            - NN*v2ok_)*v_f*dx - C2*P_/k_*v_f*dx
        
        return Fv2 + Ff
