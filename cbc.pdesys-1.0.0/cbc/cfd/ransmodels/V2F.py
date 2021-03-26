__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-09-06"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"
"""

    V2F turbulence models

"""
from TurbSolver import *
from cbc.cfd.tools.Eikonal import Eikonal
from cbc.cfd.tools.Wall import QWall

class V2F(TurbSolver):
    """
    Base class for V2F turbulence models
    """                
    def define(self):
        """define derived quantities for V2F model."""
        V,  NS = self.V['dq'], self.problem.NS_solver # Short forms        
        DQ, DQ_NoBC = DerivedQuantity, DerivedQuantity_NoBC
        NS.pdesubsystems['derived quantities'] = [
            DQ_NoBC(vars(NS), 'Sij', NS.S, "epsilon(u_)", bounded=False),
            DQ_NoBC(vars(NS), 'S2', V, 'inner(Sij_, Sij_)')]
        self.Sij_ = NS.Sij_
        self.S2_ = NS.S2_
        ns = vars(self)
        self.pdesubsystems['derived quantities'] = [
            DQ (ns, 'v2ok', V, "v2_/k_"),
            DQ_NoBC(ns, 'T' , V, "max_(min_(0.6/(Cmu*v2ok_*sqrt(6.*S2_)), k_*(1./e_)), 6.*sqrt(nu*(1./e_)))"),
            DQ_NoBC(ns, 'L' , V, "CL*max_(Ceta*(nu**3/e_)**(0.25), k_**(1.5)*min_(1./e_, 1./(Cmu*v2_*sqrt(6.*S2_))))"),
            Ce1(ns, 'Ce1', V, "1.4*(1. + Ced*sqrt(k_/v2_))"),
            DQ (ns, 'nut', V, "Cmu*v2_*T_"),
            DQ (ns, 'P', V, "2.*inner(Sij_, grad(u_))*nut_", bounded=False)
        ]
        if self.prm.get('NLV2F'): self.define_NLV2F()
        
        TurbSolver.define(self)
            
    def model_parameters(self):
        """Parameters for the V2F model."""
        model = self.prm['model']
        info('Setting parameters for %s V2F model ' %(model))
        self.problem.NS_solver.prm['apply']['S2'] = 'project'
        for dq in ['T', 'L', 'Ce1']:
            # Specify projection as default
            # (remaining DQs are use_formula by default)
            self.prm['apply'][dq] = self.prm['apply'].get(dq, 'project')
            
        self.model_prm = dict(
            Cmu = Constant(0.22, cell=self.V['dq'].cell()),
            Ce2 = Constant(1.9),
            C1 = Constant(1.4),
            C2 = Constant(0.3),
            sigma_e = Constant(1.3),
            sigma_k = Constant(1.0),
            sigma_v2 = Constant(1.0),
            e_d = Constant(0.5)
        )
        self.model_prm.update(
            dict(
                OriginalV2F=dict(
                    Ced = Constant(0.045),
                    CL = Constant(0.25),
                    Ceta = Constant(80.),
                    NN = Constant(0.)),
                LienKalizin=dict(
                    Ced = Constant(0.05),
                    CL = Constant(0.23),
                    Ceta = Constant(70.),
                    NN = Constant(5.))
                )[model]
            )
        self.__dict__.update(self.model_prm)

    def create_BCs(self, bcs):
        # Compute distance to nearest wall
        self.distance = Eikonal(self.mesh, self.boundaries)
        self.y = self.distance.y_
        return TurbSolver.create_BCs(self, bcs)
        
    # Use a nonlinear V2F model
    def define_NLV2F(self):
        DQ, DQ_NoBC = DerivedQuantity, DerivedQuantity_NoBC
        V, NS = self.V['dq'], self.problem.NS_solver
        self.RS = TensorFunctionSpace(NS.V['u'].mesh(), 'CG', self.prm['degree']['u'])
        NS.prm['apply']['Wij'] = NS.prm['apply'].get('Wij', 'project')
        # Neccessary to project eta12_ for ufl to accept it in beta1_:
        self.prm['apply']['eta12'] = 'project' 
        self.prm['apply']['rij'] = self.prm['apply'].get('rij', 'project')
        self.prm['apply']['Rij'] = self.prm['apply'].get('Rij', 'project')
        NS.pdesubsystems['derived quantities'] += [
            DQ_NoBC(vars(NS), 'Wij', self.RS, "0.5*(grad(u_) - grad(u_).T)", bounded=False)]
        self.Wij_ = NS.Wij_
        self.dij_ = Identity(NS.u_.cell().d)
        ns = vars(self)
        self.pdesubsystems['derived quantities'] += [
            DQ     (ns, 'snu', V, '2./3. - v2ok_'),
            DQ_NoBC(ns, 'eta1', V, 'T_**2*S2_'),
            DQ_NoBC(ns, 'eta2', V, 'T_**2*inner(Wij_, Wij_)'),
            DQ_NoBC(ns, 'eta12', V, 'dot(eta1_, eta2_)'),
            DQ_NoBC(ns, 'beta1', V, '1./(0.1 + sqrt(eta12_))'),
            DQ_NoBC(ns, 'gamma1', V, '1./(0.1 + eta1_)'),
            DQ_NoBC(ns, 'Cmu2', V, '6./5.*sqrt(1. - 2.*(Cmu*v2ok_)**2*eta1_)/(beta1_ + sqrt(eta12_))'),
            DQ_NoBC(ns, 'Cmu3', V, '6./5./(gamma1_ + eta1_)'),
            DQ     (ns, 'rij', NS.S, '-snu_*k_*T_**2*(Cmu2_*(Sij_*Wij_ - Wij_*Sij_) - Cmu3_*(Sij_*Sij_ - 1./3.*S2_*dij_) )', bounded=False),
            Rij    (ns, 'Rij', NS.S, "2./3.*k_*dij_ - 2.*nut_*Sij_  + rij_", bounded=False)]
        
        NS.f = self.problem.body_force() - div(self.rij_)
        
        # Note that Rij in 2D only is a (2, 2) matrix, not containing the nonzero Rij[2, 2].
        # rij[2, 2] is also nonzero, but the term is not used by Navier-Stokes in 2D and can be neglected
        # However, to get the correct trace of Rij one needs to add neglected terms. This means that 
        # k_ = 0.5*(tr(Rij_) + 2./3.*k_ - 1./3.*snu_*k_*T_**2*Cmu3_*S2_)

############ Derived quantities that require some overloading

class Ce1(DerivedQuantity):
    """Derived quantity Ce1 is using a wall function boundary condition."""    
    def create_BCs(self, bcs):
        bcu=[]
        dim = self.V.dim()
        if len(self.solver_namespace['system_names']) == 1:
            k_vector = self.solver_namespace['x_']['kev2f'][:dim]
            v2_vector = self.solver_namespace['x_']['kev2f'][2*dim:3*dim]
        elif len(self.solver_namespace['system_names']) == 2:
            k_vector = self.solver_namespace['x_']['ke'][:dim]
            v2_vector = self.solver_namespace['x_']['v2f'][:dim]
        for bc in bcs:
            if bc.type() == 'Periodic':
                bcu.append(PeriodicBC(self.V, bc))
                bcu[-1].type = bc.type
            elif bc.type() == 'Wall':
                bcu.append(QWall['Ce1'](bc, self.V, k_vector, v2_vector,
                           self.solver_namespace['Ced'](0)))
                bcu[-1].type = bc.type
        return bcu
            
class Rij(DerivedQuantity):
    
    def update(self):
        """uu, vv and ww are >= 0. Off diagonals are not."""
        N = self.V.sub(0).dim()
        dim = self.V.cell().d
        xa = self.x.array()
        start = 0
        for i in range(dim):
            stop = start + N
            xa[start:stop] = maximum(1.e-12, xa[start:stop])
            start = stop + N*(dim - 1 + i) # Symmetric
        self.x.set_local(xa)
