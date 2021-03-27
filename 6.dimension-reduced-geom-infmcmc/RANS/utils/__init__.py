# Copyright (c) 2016, The University of Texas at Austin & University of
# California, Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

#Add path to hippylib
import os
abspath = os.path.dirname( os.path.abspath(__file__) )
hippylib_path = os.path.join(abspath, "../../")
import sys
sys.path.append(abspath)
sys.path.append( hippylib_path )
print(sys.path)


# Non linear solvers
from nse_non_linear_problem import NonlinearProblem
from nse_residual_norms import ResNormAinv, ResNorml2
from nse_NewtonBacktrack import NewtonBacktrack

#Quantity of Interest:
from qoi import QOI, qoiVerify
from reduced_qoi import ReducedQOI, ReducedHessianQOI, reducedQOIVerify
from taylor_approx_qoi import TaylorApproximationQOI, plotEigenvalues

#Prior
from prior import FiniteDimensionalGaussianPrior

#Forward Propagation of uncertainty
from varianceReductionMC import varianceReductionMC

#helpers
from helpers_meshing import GradingFunctionLin, Remap
from helpers_mesh_metric import mesh_metric, h_u, hinv_u, h_over_u, h_dot_u, h_u2

#Turbulent non-reactive jet utilities
from RANS_geometry import FreeJet_Geometry, FreeJetSponge_Geometry
from RANS_loadDNSdata import loadDNSData
from RANS_model_base import RANSModel_base
from RANS_model_inadeguate_Cmu import RANSModel_inadeguate_Cmu
# from RANSProblem import RANSProblem
from RANSProblem_solcnt import RANSProblem
from RANS_qoi import JetThicknessQOI, JetThicknessQOIFSE
from RANS_misfitFunctionals import DNSDataMisfit, DNSVelocityMisfit, AlgModelMisfit, AlgModelMisfitFSE
from RANS_pp import spreadFunction, selfsimilarprofiles

from RANS_alg_model import RANS_AlgModel
from RANS_alg_model_full import RANS_AlgModelFull
from RANS_alg_model_stoc import RANS_AlgModelStoc
from RANS_alg_model_Problem import RANS_AlgModel_Problem
from RANS_alg_model_ProblemFSE import RandFieldStruct, RANS_AlgModel_ProblemFSE
