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
# from geom import geom

# MCMC
import sys
sys.path.append( "../" )
from sampler.DILI_dolfin import DILI

np.set_printoptions(precision=3, suppress=True)
np.random.seed(2017)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('propNO', nargs='?', type=int, default=1)
    parser.add_argument('num_samp', nargs='?', type=int, default=10000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=10000)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[.1,.2]) # SNR10: [.5,1.]; SNR100: [.1,.2]
#     parser.add_argument('step_nums', nargs='?', type=int, default=[1])
    parser.add_argument('props', nargs='?', type=str, default=['LI_prior', 'LI_Langevin'])
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
    print("Preparing DILI sampler using {} proposal with step sizes {} for LIS and CS resp....".format(args.props[args.propNO],args.step_sizes,))

    dili=DILI(unknown,elliptic,args.step_sizes,proposal=args.props[args.propNO],n_lag=100,n_max=100)
    dili.adaptive_MCMC(args.num_samp,args.num_burnin,threshold_l=1e-3,threshold_g=1e-3)

    # append PDE information including the count of solving
    filename_=os.path.join(dili.savepath,dili.filename+'.pckl')
    filename=os.path.join(dili.savepath,'Elliptic_'+dili.filename+'.pckl') # change filename
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
