# Utils folder

This folder contains common classes and functions that are used by different applications.
This include:
- General helpers routines: helpers_*
- Nonlinear solvers: nse_*
- Tools to manipulate quantities of interest: qoi, reduced_qoi, taylor_approx_qoi, varianceReductionMC
- Model descriptors for the RANS equations: RANS_*

## Note on copyright:

All files that implement general purpose algorithms not explicitly related to the turbulent combustion problems (RANS, Laminar flame, etc) come with the hIPPYlib copyright and they are intended to be included in a future release of hIPPYlib.

All files that contains algorithms that are problem specific to the turbulent model do not have the hIPPYlib
copyright and are not intended to be released as part of hIPPYlib.

You will mantain the copyright of every new file/model/algorithm that you create. Your contributions will not be included in hIPPYlib without your permission.

If you modify files already marked with the hIPPYlib copyright or if you wish to contribute new algorithms/models to hIPPYlib, you will be acknowledged (with your permission) as a hIPPYlib contributor.