"""
Utility functions
Shiwei Lan @ U of Warwick, 2016
Created July 25, 2016
"""

# dolfin related functions
try:
    # plot Library
    from .matplot4dolfin import matplot4dolfin
    from .dolfin_gadget import *
    from .Eigen import *
except (ImportError, RuntimeError):
    pass

# function to calculate effective sample size (ESS)
from .bayesianStats import effectiveSampleSize as ess

# some handy functions to manipulate scr matrix
from .sparse_geeklet import *