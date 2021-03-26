"""
Main function to run inverse RANS model to generate posterior samples
Shiwei Lan @ U of Warwick, 2016; @ Caltech, Sept. 2016
"""

# modules
import os,argparse,pickle
import fenics_adjoint as dl
import numpy as np

# the inverse problem
from RANS import RANS

# MCMC
import sys
sys.path.append( "../../" )
from sampler.geoinfMC_hippy import geoinfMC

np.set_printoptions(precision=3, suppress=True)
# np.random.seed(2017)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algNO', nargs='?', type=int, default=3)
    parser.add_argument('seedNO', nargs='?', type=int, default=2017)
    parser.add_argument('num_samp', nargs='?', type=int, default=1000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=0)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[.02,.3,.15,4.5,1.5])
    parser.add_argument('step_nums', nargs='?', type=int, default=[1,1,4,1,4])
    parser.add_argument('algs', nargs='?', type=str, default=('pCN','infMALA','infHMC','DRinfmMALA','DRinfmHMC'))
    args = parser.parse_args()

    # set the (global) random seed
    np.random.seed(args.seedNO)
    print('Random seed is set to %d.' % args.seedNO)
    
    # define the model
    nozz_w=1.;nx=40;ny=80
    rans=RANS(nozz_w=nozz_w,nx=nx,ny=ny)
    rans.setup(args.seedNO,src4init='solution')
    
    # (randomly) initialize parameter
#     noise = dl.Vector()
#     rans.prior.init_vector(noise,"noise")
#     Random.normal(noise, 1., True)
    PARAMETER=1
#     parameter = rans.model_stat.generate_vector(PARAMETER)
#     rans.prior.sample(noise,parameter)
    # read from MAP
    parameter = dl.Function(rans.Vh[PARAMETER])
    MAP_file=os.path.join(os.getcwd(),'analysis-L25W8-nofreshinit-yesparacont-2000/_samp_DRinfmMALA_dim3321_2018-06-06-16-24-32.h5')
    if os.path.isfile(MAP_file):
        f=dl.HDF5File(rans.mpi_comm,MAP_file,"r")
        f.read(parameter,'sample_{0}'.format(999))
        f.close()
    else:
        parameter=rans.get_MAP(SAVE=True)
    parameter=parameter.vector()
    
    # forward solver: whether do continuation
    fresh_init=False; para_cont=True
    if rans.rank == 0:
        print("Forward solving with"+{True:"",False:"out"}[fresh_init]+" repeated fresh initialization and with"+{True:"",False:"out"}[para_cont]+" parameter continuation...")

    # run MCMC to generate samples
    if rans.rank == 0:
        print("Preparing %s sampler with "+{True:"initial",False:"fixed"}[adpt_stepsz]+" step size %g for %d step(s)..."
              % (args.algs[args.algNO],args.step_sizes[args.algNO],args.step_nums[args.algNO]))

    inf_MC=geoinfMC(parameter,rans,args.step_sizes[args.algNO],args.step_nums[args.algNO],args.algs[args.algNO],True, \
                    target_acpt=0.7,fresh_init=fresh_init,para_cont=para_cont,src4init='solution')
    inf_MC.setup(args.num_samp,args.num_burnin,1,mpi_comm=rans.mpi_comm)
    if rans.rank != 0:
        inf_MC.parameters['print_level']=-1
    inf_MC.sample(num_retry_bad=0)
    
    # append PDE information including the count of solving
    filename=os.path.join(inf_MC.savepath,inf_MC.filename+'.pckl')
    f=open(filename,'ab')
#     soln_count=rans.soln_count.copy()
    soln_count=[rans.pde.solveFwd.count,rans.pde.solveAdj.count,rans.pde.solveIncremental.count]
    pickle.dump([nozz_w,nx,ny,fresh_init,para_cont,soln_count,args],f)
    f.close()
    # rename
    f_newname=os.path.join(inf_MC.savepath,'RANS_seed'+str(args.seedNO)+'_'+inf_MC.filename+'.pckl') # change filename
    if rans.rank == 0:
        os.rename(filename, f_newname)
#     # verify with load
#     f=open(filename,'rb')
#     mc_samp=pickle.load(f)
#     pde_info=pickle.load(f)
#     f.close
#     print(pde_cnt)

if __name__ == '__main__':
    main()
