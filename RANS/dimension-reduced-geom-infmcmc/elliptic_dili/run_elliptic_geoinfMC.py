"""
Main function to run Elliptic PDE model (DILI; Cui et~al, 2016) to generate posterior samples
Shiwei Lan @ Caltech, 2016
"""

# modules
import os,argparse,pickle
import numpy as np
import dolfin as df

# the inverse problem
from Elliptic_dili import Elliptic

# MCMC
import sys
sys.path.append( "../" )
from sampler.geoinfMC_dolfin import geoinfMC

np.set_printoptions(precision=3, suppress=True)
np.random.seed(2017)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algNO', nargs='?', type=int, default=0)
    parser.add_argument('num_samp', nargs='?', type=int, default=2000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=500)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[.5,2.,1.3,6.,4.]) # SNR10: [.5,2,1.3,6.,4.];SNR100: [.01,.04,0.04,.52,.25]
    parser.add_argument('step_nums', nargs='?', type=int, default=[1,1,4,1,4,1,4])
    parser.add_argument('algs', nargs='?', type=str, default=('pCN','infMALA','infHMC','DRinfmMALA','DRinfmHMC'))
    args = parser.parse_args()

    ## define the inverse elliptic problem ##
    # parameters for PDE model
    nx=40;ny=40;
    # parameters for prior model
    sigma=1.25;s=0.0625
    # parameters for misfit model
    SNR=10 # 100
    # define the inverse problem
    elliptic=Elliptic(nx=nx,ny=ny,SNR=SNR,sigma=sigma,s=s)
    
    # initialization
#     unknown=elliptic.prior.sample(whiten=False)
    unknown=elliptic.prior.gen_vector()
#     unknown=df.Function(elliptic.pde.V)
#     MAP_file=os.path.join(os.getcwd(),'result/MAP.h5')
#     if os.path.isfile(MAP_file):
#         f=df.HDF5File(elliptic.pde.mpi_comm,MAP_file,"r")
#         f.read(unknown,'parameter')
#         f.close()
#     else:
#         unknown=elliptic.get_MAP(SAVE=True)
    
    # run MCMC to generate samples
    print("Preparing %s sampler with step size %g for %d step(s)..."
          % (args.algs[args.algNO],args.step_sizes[args.algNO],args.step_nums[args.algNO]))
    
    inf_GMC=geoinfMC(unknown,elliptic,args.step_sizes[args.algNO],args.step_nums[args.algNO],args.algs[args.algNO],k=5)
    mc_fun=inf_GMC.sample
    mc_args=(args.num_samp,args.num_burnin)
    mc_fun(*mc_args)
    
    # append PDE information including the count of solving
    filename_=os.path.join(inf_GMC.savepath,inf_GMC.filename+'.pckl')
    filename=os.path.join(inf_GMC.savepath,'Elliptic_'+inf_GMC.filename+'.pckl') # change filename
    os.rename(filename_, filename)
    f=open(filename,'ab')
#     soln_count=[elliptic.soln_count,elliptic.pde.soln_count]
    soln_count=elliptic.pde.soln_count
    pickle.dump([nx,ny,sigma,s,SNR,soln_count,args],f)
    f.close()
#     # verify with load
#     f=open(filename,'rb')
#     mc_samp=pickle.load(f)
#     pde_info=pickle.load(f)
#     f.close
#     print(pde_cnt)

if __name__ == '__main__':
    main()
