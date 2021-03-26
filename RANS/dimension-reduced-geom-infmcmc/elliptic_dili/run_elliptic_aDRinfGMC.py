"""
Main function to run Elliptic PDE model (DILI; Cui et~al, 2016) to generate posterior samples
Shiwei Lan @ U of Warwick, 2016
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
# from sampler.aDRinfmMALA_dolfin import aDRinfmMALA
# from sampler.aDRinfmHMC_dolfin import aDRinfmHMC
from sampler.aDRinfGMC_dolfin import aDRinfGMC

# sys.path.append( "/home/fenics/shared/pysrc/" )
# import pydevd
# pydevd.settrace('10.2.15.181', port=5678)

np.set_printoptions(precision=3, suppress=True)
np.random.seed(2017)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algNO', nargs='?', type=int, default=0)
    parser.add_argument('propNO', nargs='?', type=int, default=1)
    parser.add_argument('num_samp', nargs='?', type=int, default=10000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=10000)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[.25,.1]) # SNR10: [3.,1.5]; SNR100: [.22,.06]
    parser.add_argument('step_nums', nargs='?', type=int, default=[1,4])
    parser.add_argument('algs', nargs='?', type=str, default=('aDRinfmMALA','aDRinfmHMC'))
    parser.add_argument('props', nargs='?', type=str, default=('LI_prior', 'LI_Langevin'))
    args = parser.parse_args()


    ## define the inverse elliptic problem ##
    # parameters for PDE model
    nx=40;ny=40;
    # parameters for prior model
    sigma=1.25;s=0.0625
    # parameters for misfit model
    SNR=100 # 100
    # define the inverse problem
    elliptic=Elliptic(nx=nx,ny=ny,SNR=SNR,sigma=sigma,s=s)
    
    # initialization
#     unknown=elliptic.prior.sample(whiten=True)
    unknown=elliptic.prior.gen_vector()
#     unknown=df.Function(elliptic.pde.V)
#     MAP_file=os.path.join(os.getcwd(),'result/MAP_SNR'+str(SNR)+'.h5')
#     if os.path.isfile(MAP_file):
#         f=df.HDF5File(elliptic.pde.mpi_comm,MAP_file,"r")
#         f.read(unknown,'parameter')
#         f.close()
#     else:
#         unknown=elliptic.get_MAP(SAVE=True)
#     unknown=elliptic.prior.u2v(unknown.vector())
    
    # run MCMC to generate samples
    print("Preparing %s sampler using %s proposal with step size %g for %d step(s)..."
          % (args.algs[args.algNO],args.props[args.propNO],args.step_sizes[args.algNO],args.step_nums[args.algNO]))

#     if args.algs[args.algNO] is 'aDRinfmMALA':
#         adrinfmGMC=aDRinfmMALA(unknown,elliptic,args.step_size,proposal=args.props[args.propNO])
#     elif args.algs[args.algNO] is 'aDRinfmHMC':
#         adrinfmGMC=aDRinfmHMC(unknown,elliptic,args.step_size,args.step_nums[args.algNO],proposal=args.props[args.propNO])
#     else:
#         print('Algorithm not available!')
#         raise
    adrinfmGMC=aDRinfGMC(unknown,elliptic,args.step_sizes[args.algNO],args.step_nums[args.algNO],args.algs[args.algNO],args.props[args.propNO],n_lag=100,n_max=100)
    adrinfmGMC.adaptive_MCMC(args.num_samp,args.num_burnin,threshold_l=1e-3,threshold_g=1e-3)

    # append PDE information including the count of solving
    filename_=os.path.join(adrinfmGMC.savepath,adrinfmGMC.filename+'.pckl')
    filename=os.path.join(adrinfmGMC.savepath,'Elliptic_'+adrinfmGMC.filename+'.pckl') # change filename
    os.rename(filename_, filename)
    f=open(filename,'ab')
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
