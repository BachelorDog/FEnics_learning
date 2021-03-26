#!/bin/bash
##PBS -N rans_mcmc
##PBS -q default
##PBS -l nodes=1:ppn=12,walltime=100:00:00
##PBS -V
##PBS -m bae -M slan@caltech.edu

# load FEniCS environment
source ${HOME}/FEniCS-1.6.0/fenics_1.6.0.sh

# go to working directory
# cd ~/FIS-MCMC/code/RANS/RANS_bip

# run python script
if [ $# -eq 0 ]; then
	mc_name='geoinfMC'
	alg_NO=0
	seed_NO=2017
	n_cores=30
elif [ $# -eq 1 ]; then
	mc_name="$1"
	alg_NO=0
	seed_NO=2017
	n_cores=30
elif [ $# -eq 2 ]; then
	mc_name="$1"
	alg_NO="$2"
	seed_NO=2017
	n_cores=30
elif [ $# -eq 3 ]; then
	mc_name="$1"
	alg_NO="$2"
	seed_NO="$3"
	n_cores=30
elif [ $# -eq 4 ]; then
	mc_name="$1"
	alg_NO="$2"
	seed_NO="$3"
	n_cores="$4"
fi
prop_NO=1

if [[ ${mc_name}=='geoinfMC' ]]; then
	mpirun -np ${n_cores} python -u run_RANS_${mc_name}.py ${alg_NO} ${seed_NO}
elif [[ ${mc_name}=='dili' ]]; then
	mpirun -np ${n_cores} python -u run_RANS_${mc_name}.py ${prop_NO} ${seed_NO}
elif [[ ${mc_name}=='aDRinfGMC' ]]; then
	mpirun -np ${n_cores} python -u run_RANS_${mc_name}.py ${alg_NO} ${prop_NO} ${seed_NO}
else
	echo "Wrong args!"
	exit 0
fi

