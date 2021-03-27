"""
Main function to run Elliptic PDE model (DILI; Cui et~al, 2016) to generate posterior samples
Shiwei Lan @ U of Warwick, 2016
"""

import os,argparse,pickle
from dolfin import *
import numpy as np
# PDE
from Elliptic import Elliptic
from geom import geom

# MCMC
# from ..sampler.geoinfMC import geoinfMC
import sys
sys.path.insert(0,'../sampler')
from geoinfMC import geoinfMC

np.set_printoptions(precision=3, suppress=True)
np.random.seed(2016)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algNO', nargs='?', type=int, default=3)
    parser.add_argument('dim', nargs='?', type=int, default=100)
    parser.add_argument('num_samp', nargs='?', type=int, default=10000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=1000)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[1,10,1,5,100,100,100]) # 1,20,2,100,100,100,100
    parser.add_argument('step_nums', nargs='?', type=int, default=[1,1,4,1,4,1,4])
    parser.add_argument('algs', nargs='?', type=str, default=('pCN','infMALA','infHMC','infmMALA','infmHMC','splitinfmMALA','splitinfmHMC'))
    parser.add_argument('trct_opt', nargs='?', type=int, default=3)
    parser.add_argument('trct_idx', nargs='?', type=int, default=[])
    args = parser.parse_args()


    # define the PDE problem
    nx=40;ny=40;
    elliptic=Elliptic(nx=nx,ny=ny)
    sigma=1.25;s=0.0625
    kl_opt='kf' # choice (kernel function) for Karhunen-Loeve expansion of coefficient function
    theta=np.zeros(args.dim) # initialization
#     f_kereigen=os.path.join(os.getcwd(),'kernel_eigens.pckl')
#     if os.path.isfile(f_kereigen) and os.access(f_kereigen, os.R_OK):
#         f=open(f_kereigen,'rb')
#         ker,eigen=pickle.load(f)
#         f.close()
#     else:
#         ker=elliptic.kernel(sigma=sigma,s=s)
#         eigen=ker.get_eigen(args.dim)
#         f=open(f_kereigen,'wb')
#         pickle.dump([ker,eigen],f)
#         f.close()
#     coeff=elliptic.coefficient(theta=theta,kl_opt=kl_opt,sigma=sigma,s=s,degree=2,ker=ker,eigen=eigen) # cannot pickle SwigPyObject??
    coeff=elliptic.coefficient(theta=theta,kl_opt=kl_opt,sigma=sigma,s=s,degree=2)

    # obtain observations
    SNR=100 # 50
    f_obs=os.path.join(os.getcwd(),'obs_'+str(SNR)+'.pckl')
    if os.path.isfile(f_obs):
        f=open(f_obs,'rb')
        obs,idx,loc,sd_noise=pickle.load(f)
        f.close()
    else:
        obs,idx,loc,sd_noise=elliptic.get_obs(SNR=SNR)
        f=open(f_obs,'wb')
        pickle.dump([obs,idx,loc,sd_noise],f)
        f.close()

    # define data misfit class
    print('Defining data-misfit...')
    misfit=elliptic.data_misfit(obs,1./sd_noise**2,idx,loc)


    # run MCMC to generate samples
    print("Preparing %s sampler with step size %g for %d step(s)..."
          % (args.algs[args.algNO],args.step_sizes[args.algNO],args.step_nums[args.algNO]))

    if 'split' in args.algs[args.algNO]:
        args.trct_opt = 2
        args.trct_idx = range(25)
        print('and truncating on '+{0:'value, gradient, and metric (no sense)',
                                    1:'gradient and metric',
                                    2:'metric',
                                    3:'none'}[args.trct_opt]+'...')

    geomfun=lambda theta,geom_opt: geom(theta,coeff,elliptic,misfit,geom_opt,args.trct_opt,args.trct_idx)
    inf_MC=geoinfMC(theta,np.eye(args.dim),geomfun,args.step_sizes[args.algNO],args.step_nums[args.algNO],args.algs[args.algNO],args.trct_idx)
    mc_fun=inf_MC.sample
    mc_args=(args.num_samp,args.num_burnin)
    mc_fun(*mc_args)

    # append PDE information including the count of solving
    filename_=os.path.join(inf_MC.savepath,inf_MC.filename+'.pckl')
    filename=os.path.join(inf_MC.savepath,'Elliptic_'+inf_MC.filename+'.pckl') # change filename
    os.rename(filename_, filename)
    f=open(filename,'ab')
    soln_count=elliptic.soln_count.copy()
    pickle.dump([nx,ny,theta,sigma,s,SNR,sd_noise,soln_count,args],f)
    f.close()
#     # verify with load
#     f=open(filename,'rb')
#     mc_samp=pickle.load(f)
#     pde_info=pickle.load(f)
#     f.close
#     print(pde_cnt)

if __name__ == '__main__':
    main()
