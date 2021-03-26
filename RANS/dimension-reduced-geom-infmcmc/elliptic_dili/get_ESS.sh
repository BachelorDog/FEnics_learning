#!/usr/bin/env bash -l

# load FEniCS environment
source ${HOME}/FEniCS/fenics.sh

# run python script to get ESS of samples stored in h5 format
python -u get_ESS_dolfin.py