"""
Combine MCMC samples
Shiwei Lan @ CalTech, 2018
"""

import os
import dolfin as dl
import numpy as np

# from joblib import Parallel, delayed

# def restore_each_sample(f,samp_f,s):
#     f.read(samp_f,'sample_{0}'.format(s))
#     return samp_f.vector()

def restore_sample(mpi_comm,V,dir_name,f_name,num_samp):
    f=dl.HDF5File(mpi_comm,os.path.join(dir_name,f_name),"r")
    samp_f=dl.Function(V,name="parameter")
    samp=np.zeros((num_samp,V.dim()))
    bad_idx=[]
    prog=np.ceil(num_samp*(.1+np.arange(0,1,.1)))
    for s in xrange(num_samp):
        try:
            f.read(samp_f,'sample_{0}'.format(s))
        except:
#             print('Bad sample encountered!')
            bad_idx.append(s)
        samp[s,]=samp_f.vector()
        if s+1 in prog:
            print('{0:.0f}% samples have been restored.'.format(np.float(s+1)/num_samp*100))
#     f_read=lambda s: restore_each_sample(f,samp_f,s)
#     samp=Parallel(n_jobs=4)(delayed(f_read)(i) for i in range(num_samp))
    f.close()
    if len(bad_idx)>0:
        print('{0:d} bad samples encountered!'.format(len(bad_idx)))
        samp=np.delete(samp,bad_idx,0)
    return samp

if __name__ == '__main__':
    from RANS import RANS
    import sys
    sys.path.append( "../" )
    import pickle
    
    # define the PDE problem
    nozz_w=1.;nx=40;ny=80
    rans=RANS(nozz_w=nozz_w,nx=nx,ny=ny)
    seed_NO=2017
    rans.setup(seed=seed_NO)
    # algorithms
    algs=('DRinfmMALA',)
    alg_names=('DR-$\infty$-mMALA',)
    num_algs=len(algs)
    # preparation for estimates
    folder = './analysis-L25W8-nofreshinit-yesparacont-2000'
#         
    found=np.zeros(num_algs,dtype=bool)
    for a in range(num_algs):
        print('Working on '+alg_names[a]+' algorithm...')
        
#         # combine samples
#         fnames=[f for f in os.listdir(folder) if f.endswith('.h5')]
#         num_samp=2000; _samp=[]
#         for f_i in fnames:
#             if '_samp_'+algs[a]+'_' in f_i:
#                 try:
#                     samp=restore_sample(rans.mpi_comm,rans.Vh_PARAMETER,folder,f_i,num_samp/2)
#                     _samp.append(samp)
#                     found[a]=True
#                     print(f_i+' processed.\n')
#                 except:
#                     pass
#         if found[a]:
#             _samp=np.concatenate(tuple(_samp))
#             num_samp=_samp.shape[0]
#             # store combined samples in HDF5 file
#             f_name='_samp_'+algs[a]+'_dim'+str(samp.shape[1])+'_combined.h5'
#             f=dl.HDF5File(rans.mpi_comm,os.path.join(folder,f_name),"w")
#             samp_f=dl.Function(rans.Vh_PARAMETER,name="parameter")
#             for i in xrange(num_samp):
#                 samp_f.vector()[:]=_samp[i,:]
#                 f.write(samp_f,'sample_{0}'.format(i))
#             f.close()
#             print(f_name+' generated.')
         
        # combine other statistics
        fnames=[f for f in os.listdir(folder) if f.endswith('.pckl')]
        _loglik=[];_acpt=[];_time=[];_soln_count=[];_num_samp=[]
        found_a=False
        for f_i in fnames:
            if '_'+algs[a]+'_' in f_i:
                try:
                    f=open(os.path.join(folder,f_i),'rb')
                    try:
                        f_read1=pickle.load(f)
                    except:
                        f_read1=pickle.load(f,encoding='bytes')
                    loglik,acpt,time=f_read1[3:6]
                    _loglik.append(loglik); _acpt.append(acpt); _time.append(time)
                    try:
                        f_read2=pickle.load(f)
                        soln_count,args=f_read2[5:7]
                        num_samp=int(str(args).split("num_samp=")[-1].split(",")[0])
                        _soln_count.append(soln_count); _num_samp.append(num_samp)
                    except:
                        print('pickle file broken!')
                    f.close()
                    found_a=True
                    dim=f_i.split("dim")[-1].split("_")[0]
                    print(f_i+' processed.\n')
                except:
                    print('Need manual input.')
        if found_a:
            _loglik=np.concatenate(tuple(_loglik))
            _acpt=np.mean(_acpt)
            _time=np.sum(_time)
            _soln_count=np.sum(_soln_count,axis=0,dtype=np.int); _num_samp=np.sum(_num_samp,dtype=np.int)
            # store combined results
            f_name='RANS_seed'+str(seed_NO)+'_'+algs[a]+'_dim'+str(dim)+'_combined.pckl'
            f=open(os.path.join(folder,f_name),'ab')
            f_read1[3:6]=_loglik,_acpt,_time
            pickle.dump(f_read1,f)
            if _num_samp>0:
                f_read2[5:7]=_soln_count,_num_samp
                pickle.dump(f_read2,f)
            f.close()
            print(f_name+' generated.')
    