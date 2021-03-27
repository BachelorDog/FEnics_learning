__author__ = "Jorgen Myre <jorgenmy@math.uio.no>"
__date__ = "2011-02-22"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU GPL version 3 or any later version?"
"""

    K-Epsilon/Reynolds-stress turbulence models

"""
from TurbSolver import *
from cbc.cfd.tools.Eikonal import Eikonal
from cbc.cfd.tools.Wall import QWall #Need to make new wall-function to match system

class ER(TurbSolver):
    """
    Base class for ER turbulence models
    NOTE:  Rij and Fij should be stored as variables, not DQs, unlike in normal model
    """                
    
    def __init__(self, system_composition, problem, parameters):
        # A segregated system of two coupled systems:
        parameters['space']['Rij'] = TensorFunctionSpace
        parameters['space']['Fij'] = TensorFunctionSpace
        self.dim = problem.NS_problem.mesh.geometry().dim()
        # When symmetric tensors is possible:
        #parameters['symmetry']['Rij'] = dict(((i,j), (j,i)) 
        #    for i in range(self.dim) for j in range(self.dim) if i > j )
        #parameters['symmetry']['Fij'] = dict(((i,j), (j,i)) 
        #    for i in range(self.dim) for j in range(self.dim) if i > j )        
        TurbSolver.__init__(self,
                     system_composition=system_composition,
                     problem=problem,
                     parameters=parameters)
                     
    def define(self):
        """define derived quantities for ER model."""
        V,  NS = self.V['dq'], self.problem.NS_solver # Short forms        
        DQ, DQ_NoBC = DerivedQuantity, DerivedQuantity_NoBC
        NS.V['SS'] = TensorFunctionSpace(self.problem.mesh, 
                          self.prm['family']['Rij'], self.prm['degree']['Rij'])
                                           
        NS.pdesubsystems['derived quantities'] = [
            DQ_NoBC(vars(NS), 'Sij', NS.S, "epsilon(u_)", bounded=False, 
                    apply='project'),
            DQ_NoBC(vars(NS), 'Wij', NS.V['SS'], "0.5*(grad(u_) - grad(u_).T)",
                    bounded=False)]
        self.Sij_ = NS.Sij_
        self.Wij_ = NS.Wij_
        self.i, self.j, self.m, self.l = i, j, Index(2),  l
        self.dim = V.cell().d
        self.Aij = self.Rij*(0.5/self.k_) - 1./3.*self.dij
        ns = vars(self)
        # A33 = - (A11 + A33) = - tr(Aij),  AijAji = inner(Aij, Aij) + A33**2 = inner(Aij, Aij) + trace(Aij)**2
        self.pdesubsystems['derived quantities'] = [
            DQ_NoBC(ns, 'T' , V, "max_(k_*(1./e_), 6.*sqrt(nu*(1./e_)))"),
            DQ_NoBC(ns, 'L' , V, "CL*max_(Ceta*(nu**3/e_)**(0.25), k_**(1.5)*(1./e_))"),
            DQ_NoBC(ns, 'Aij', NS.S, "Rij_*(0.5/k_) - 1./3.*dij", bounded=False),
            DQ_NoBC(ns, 'Pij', self.V['Rij'], 
                    "- dot(Rij_, grad(u_).T) - dot(grad(u_), Rij_.T)", bounded=False),                        
            #DQ_NoBC(self, 'Ce1_', V, "1.3 + 0.25/(1. + (0.15*y/L_)**2)**4", ns, bounded=True),
            DQ_NoBC(ns, 'Ce1', V, 
             "1.4*(1. + Ced*sqrt(k_/max_(1.e-10, inner(Rij_, outer(ni, ni)))))", 
             bounded=True),
            DQ (ns, 'nut', V, 'Cmu*(inner(Rij_, outer(ni, ni)))*T_')            
        ]
        if self.problem.prm['turbulence_model'] == 'LRR-IP':
            self.pdesubsystems['derived quantities'] += [   
                DQ_NoBC(self, 'PHIij', self.V['Rij'], "-CR*(1./T_)*2.*Aij*k_ \
                                    - C2*(Pij_ - 1./3.*tr(Pij_)*dij)", ns, bounded=False)]
        
        elif self.problem.prm['turbulence_model'] == 'SSG':
            self.pdesubsystems['derived quantities'] += [   
                DQ_NoBC(self, 'PHIij', self.V['Rij'], "-( Cp1*e_ + Cp1s*0.5*tr(Pij_) )*Aij \
                                    + Cp2*e_*(dot(Aij_, Aij_) - 1./3*(inner(Aij, Aij_)+ tr(Aij_)**2)*dij ) \
                                    + ( Cp3 - Cp3s*sqrt(inner(Aij_, Aij_) + tr(Aij_)**2) )*k_*Sij_ \
                                    + Cp4*k_*(dot(Aij_, Sij_) + dot(Sij_,Aij_) - 2./3*inner(Aij_, Sij_)*dij) \
                                    + Cp5*k_*(dot(Wij_, Aij_) - dot(Aij_, Wij_) )", ns, bounded=False)]

        # insert Rij and "anti-stabilization"-term into f, similar to setting nu = nu + nut
        #NS.f = self.Turb_problem.NS_problem.body_force() - div(self.Rij_ - 2.*self.nut_*self.Sij_)
        NS.correction = self.Rij_ - 2.*self.nut_*self.Sij_ # For testing using integration by parts. Used with Steady_Coupled_5
        
        TurbSolver.define(self)
            
    def model_parameters(self):  
        """Parameters for the ER model."""
        model = self.problem.prm['turbulence_model']
        info('Setting parameters for %s ER model ' %(model))
        for dq in ['T', 'L', 'nut']:
            # Specify projection as default
            # (remaining DQs are use_formula by default)
            self.prm['apply'][dq] = self.prm['apply'].get(dq, 'project')
            
        self.model_prm = dict(
            Cmu_nut = Constant(0.09),
            Ce1 = Constant(1.44),
            Ced = Constant(0.045),
            Ce2 = Constant(1.9),
            sigma_e = Constant(1.3),
            sigma_k = Constant(1.0),
            Cp1  = Constant(3.4),
            Cp1s = Constant(1.8),
            Cp2  = Constant(4.2),
            Cp3  = Constant(0.8),
            Cp3s = Constant(1.30),
            Cp4  = Constant(1.25),
            Cp5 = Constant(0.4),
            Ceta = Constant(80.0),
            CL = Constant(0.25),
            Cmu = Constant(0.22),
            e_d = Constant(0.5),
            CR = Constant(1.8),
            C2 = Constant(3./5.)
        )
        self.dij = Identity(self.V['dq'].cell().d)
        self.__dict__.update(self.model_prm)

    def create_BCs(self, bcs):
        # Compute distance to nearest wall
        self.distance = Eikonal(self.V['dq'], self.boundaries)
        self.y = self.distance.y
        DerivedQuantity_NoBC(vars(self), 'ni', VectorFunctionSpace(self.problem.mesh, 
                             self.prm['family']['dq'], self.prm['degree']['dq']), 
                             "grad(y)/sqrt(inner(grad(y), grad(y)))", 
                             bounded=False, apply='project')
        self.ti = Function(VectorFunctionSpace(self.problem.mesh, 
                           self.prm['family']['dq'], self.prm['degree']['dq']))
        N = self.ti.vector().size()/2
        self.ti.vector()[:N] = self.ni.vector()[N:]
        self.ti.vector()[N:] = -self.ni.vector()[:N]
        
        bcu = {}
        for name in self.system_names:
            bcu[name] = []
            
        for bc in bcs:
            for name in self.system_names:
                V = self.V[name]
                if bc.type() in ('VelocityInlet', 'Wall'):
                    if hasattr(bc, 'func'):
                        if isinstance(bc.func, dict):
                            add_BC(bcu[name], V, bc, bc.func[name])
                        else:
                            add_BC(bcu[name], V, bc, bc.func)
                    else:
                        if bc.type() == 'Wall': # Default is zero on walls
                            if isinstance(V, FunctionSpace):
                                func = Constant(1e-12)
                            elif isinstance(V, (MixedFunctionSpace, 
                                                VectorFunctionSpace)):
                                func = Constant((1e-12, )*V.cell().d)
                                if V.sub(0).num_sub_spaces() > 2:
                                    func = Constant((1e-12, )*8)
                                    #func = Constant((1e-12, )*V.num_sub_spaces()*V.sub(0).num_sub_spaces())
                        
                            elif isinstance(V, TensorFunctionSpace):                                
                                func = Expression((('1.e-12', )*V.cell().d, )*
                                                  V.cell().d)
                            else:
                                raise NotImplementedError
                            add_BC(bcu[name], V, bc, func)
                        elif bc.type() == 'VelocityInlet':
                            raise TypeError('expected func for VelocityInlet')        
                elif bc.type() in ('ConstantPressure', 'Outlet'):
                    # This bc could be weakly enforced
                    bcu[name].append(bc)
                elif bc.type() == 'Periodic':
                    add_BC(bcu[name], V, bc, None)
                else:
                    info("No assigned boundary condition for %s -- skipping..."
                         %(bc.__class__.__name__))                
        return bcu

            