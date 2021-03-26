__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-09-06"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"
"""

    LowReynolds turbulence models

"""
from TurbSolver import *
from cbc.cfd.tools.Eikonal import Eikonal

class LowReynolds(TurbSolver):
    """Base class for low-Reynolds number turbulence models."""
        
    def define(self):
        """ Set up linear algebra schemes and their boundary conditions """
        V,  NS = self.V['dq'], self.problem.NS_solver 
        model = self.prm['model']
        DQ, DQ_NoBC = DerivedQuantity, DerivedQuantity_NoBC
        NS.pdesubsystems['derived quantities'] = [
            DQ_NoBC(vars(NS), 'Sij', NS.S, "0.5*(grad(u_)+grad(u_).T)",
                    bounded=False),
            DerivedQuantity_grad(vars(NS), 'd2udy2', NS.V['u'], "-grad(u_)", 
                                 bounded=False)]        
        self.Sij_ = NS.Sij_; self.d2udy2_ = NS.d2udy2_
        ns = vars(self)   # No copy. It is updated automatically.
        self.pdesubsystems['derived quantities'] = dict(
            LaunderSharma=lambda :[
                DQ_NoBC(ns, 'D', V, "nu/2.*(1./k_)*inner(grad(k_), grad(k_))"),
                DQ(ns, 'fmu', V, "exp(-3.4/(1. + (k_*k_/nu/e_)/50.)**2)", 
                   wall_value=exp(-3.4)),
                DQ(ns, 'f2', V, "1. - 0.3*exp(-(k_*k_/nu/e_)**2)", wall_value=0.7),
                DQ(ns, 'nut', V, "Cmu*fmu_*k_*k_*(1./e_)"),
                DQ(ns, 'E0', V, "2.*nu*nut_*dot(d2udy2_, d2udy2_)")],
            JonesLaunder=lambda :[
                DQ_NoBC(ns, 'D', V, "nu/2.*(1/k_)*inner(grad(k_), grad(k_))"),
                DQ(ns, 'fmu', V, "exp(-2.5/(1. + (k_*k_/nu/e_)/50.))", 
                   wall_value=exp(-2.5)),
                DQ(ns, 'f2', V, "(1. - 0.3*exp(-(k_*k_/nu/e_)**2))", 
                   wall_value=0.7),
                DQ(ns, 'nut', V, "Cmu*fmu_*k_*k_*(1./e_)"),
                DQ(ns, 'E0', V, "2.*nu*nut_*dot(d2udy2_, d2udy2_)")],
            Chien=lambda :[
                DQ_NoBC(ns, 'D', V, "2.*nu*k_/y**2"),
                DQ(ns, 'yplus', V, "y*Cmu**(0.25)*k_**(0.5)/nu"),
                DQ(ns, 'fmu', V, "(1. - exp(-0.0115*yplus_))"),
                DQ(ns, 'f2', V, "(1. - 0.22*exp(-(k_*k_/nu/e_/6.)**2))", 
                   wall_value=0.78),
                DQ(ns, 'nut', V, "Cmu*fmu_*k_*k_*(1./e_)"),
                DQ_NoBC(ns, 'E0', V, "-2.*nu*e_/y**2*exp(-yplus_/2.)",
                        bounded=False)] # minus?
            )[model]() # Note. lambda is used to delay the construction of all the DQ objects until we index by [model] and call the lambda function
        
        TurbSolver.define(self)
                
    def model_parameters(self):
        model = self.prm['model']
        info('Setting parameters for model %s' %(model))
        for dq in ('nut',):
            self.prm['apply'][dq] = self.prm['apply'].get(dq, 'project')
        self.problem.NS_solver.prm['apply']['d2udy2'] = 'project' 

        self.model_prm = dict(
            Cmu = 0.09,
            sigma_e = 1.30,
            sigma_k = 1.0,
            e_nut = 1.0,
            e_d = 0.,
            f1 = 1.0)
        Ce1_Ce2 = dict(
            LaunderSharma=dict(Ce1 = 1.44, Ce2 = 1.92),
            JonesLaunder=dict(Ce1 = 1.55, Ce2 = 2.0),
            Chien=dict(Ce1 = 1.35, Ce2 = 1.80))
        self.model_prm.update(Ce1_Ce2[model])
        # wrap in Constant objects:
        for name in self.model_prm:
            self.model_prm[name] = Constant(self.model_prm[name], cell=self.V['dq'].cell())
            
        self.__dict__.update(self.model_prm)

    def create_BCs(self, bcs):
        # Compute distance to nearest wall
        self.distance = Eikonal(self.mesh, self.boundaries)
        self.y = self.distance.y_
        return TurbSolver.create_BCs(self, bcs)
