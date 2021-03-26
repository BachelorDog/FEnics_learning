__author__ = "Jorgen Myre <jorgenmy@math.uio.no>"
__date__ = "2011-02-22"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU GPL version 3 or any later version?"
"""

    ER turbulence model
    The two systems (k and epsilon) and (Rij and Fij) are individually solved

"""
from ER import *
from cbc.cfd.tools.Wall import QWall

class ER_2Coupled(ER):
    
    def __init__(self, problem, parameters):
        
        ER.__init__(self,
                     system_composition=[['k', 'e'], ['Rij', 'Fij']],
                     problem=problem,
                     parameters=parameters)
                
    def define(self):
        ER.define(self)
        self.schemes['ke']  =  eval(self.prm['time_integration'] + '_ke_' + 
                                     str(self.prm['scheme']['ke']))(self,
                                         self.system_composition[0])
        self.schemes['RijFij'] =  eval(self.prm['time_integration'] + '_RijFij_' + 
                                       str(self.prm['scheme']['RijFij']))(self,
                                       self.system_composition[1])

    def create_BCs(self, bcs):
        # Set regular boundary conditions
        bcu = ER.create_BCs(self, bcs)
        # Set wall functions
        for bc in bcs:
            if bc.type() == 'Wall':
                bcu['ke'].append(QWall['ke'](bc, self.y, self.nu(0))) 
                bcu['ke'][-1].type = bc.type
                bcu['RijFij'].append(QWall['Fij_2'](bc, self.y, self.nu(0), self.ke_, self.ni))
                #bcu['RijFij'].append(QWall['Fij_2'](bc, self.y, self.nu(0), self.ke_))
                bcu['RijFij'][-1].type = bc.type
        return bcu

    # Updates stored variables before solving next part of the system
    def solve_inner(self, max_iter=1, max_err=1e-7, update=lambda: None,
                    logging=True):
        total_error = ""
        for name in self.system_names:
            err, j = solve_nonlinear([self.schemes[name]],
                                 max_iter=max_iter, max_err=max_err,
                                 update=update, logging=logging)
            self.solve_derived_quantities()
            total_error += err 
        self.total_number_iters += j
        return total_error
        
class KEBase(TurbModel):   
    def update(self):
        """This makes k=1e-10 on walls."""
        bound(self.x, minf=1e-10)
    
class RIJFIJBase(TurbModel):    
    def update(self):
        """uu, vv and ww are >= 0. Off diagonals are not."""
        N = self.V.sub(0).sub(0).dim()
        dim = 2
        xa = self.x.array()
        start = 0
        for i in range(dim):
            stop = start + N
            xa[start:stop] = maximum(1.e-12, xa[start:stop])
            start = stop + N*dim # unsymmetric
            
        xa[2*N:3*N] = xa[N:2*N]
        xa[6*N:7*N] = xa[5*N:6*N]
        self.x.set_local(xa)
        
class Steady_ke_1(KEBase):
    def form(self, k, e, v_k, v_e, k_, e_, Rij_, Pij_, nu, u_, Ce1, Ce1_, T_, Ce2, 
                   Cmu, e_d, sigma_e, nut_, **kwargs):
        Fk = nu*inner( grad(k) , grad(v_k) )*dx \
             + inner( dot( u_, grad(k) ) , v_k )*dx \
             - inner( 0.5*tr(Pij_) , v_k)*dx \
             + (e*e_d + k*(1./k_)*e_*(1. - e_d))*v_k*dx \
             + inner( Cmu*T_*dot(Rij_,grad(k)) , grad(v_k) )*dx
             #+ nut_*inner(grad(v_k), grad(k))*dx
             
        Fe = nu*inner( grad(e) , grad(v_e) )*dx \
             + inner( dot( u_, grad(e) ) , v_e )*dx \
             - inner( 0.5*Ce1_*(1./T_)*tr(Pij_) , v_e)*dx \
             + Ce2*(1./T_)*e*v_e*dx \
             + inner( Cmu*T_*(1./sigma_e)*dot(Rij_,grad(e)) , grad(v_e) )*dx          
             #+ nut_*(1./sigma_e)*inner(grad(v_e), grad(e))*dx
             
        return Fk + Fe

class Steady_RijFij_1(RIJFIJBase):
    def form(self, Rij, Rij_, v_Rij, k_, e_, Pij_, nu, u_, nut_,
             Fij, Fij_, v_Fij, Aij_, Aij, PHIij_, Cmu, T_, L_, **kwargs):
        Fr = nu*inner(grad(Rij), grad(v_Rij))*dx \
             + inner( dot(grad(Rij), u_) , v_Rij )*dx \
             - inner( k_*Fij , v_Rij )*dx \
             - inner( Pij_ , v_Rij )*dx \
             + inner( Rij*e_*(1./k_) , v_Rij)*dx \
             + inner( Cmu*T_*dot(grad(Rij), Rij_) , grad(v_Rij) )*dx
             #+ nut_*inner(grad(Rij) , grad(v_Rij) )*dx
             
        Ff = inner( grad(Fij), grad(L_**2*v_Fij) )*dx \
             + inner( Fij , v_Fij )*dx \
             - (1./k_)*inner( PHIij_ , v_Fij )*dx \
             - (2./T_)*inner( Aij_ , v_Fij )*dx

        return Fr + Ff
