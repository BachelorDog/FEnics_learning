"""
Analyze MCMC samples
Shiwei Lan @ U of Warwick, 2016
"""

import os,pickle
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from bayesianStats import effectiveSampleSize as ess
from joblib import Parallel, delayed

class ana_samp(object):
    def __init__(self,algs,dir_name='',ext='.pckl',save_txt=True,PLOT=False,save_fig=False):
        self.algs=algs
        self.num_algs=len(algs)
        # locate the folder
        cwd=os.getcwd()
        self.savepath=os.path.join(cwd,dir_name)
        # scan files
        self.fnames=[f for f in os.listdir(self.savepath) if f.endswith(ext)]
        # some settings
        self.save_txt=save_txt
        self.PLOT=PLOT
        self.save_fig=save_fig

    def cal_ESS(self,samp):
        num_samp,dim=np.shape(samp)
#         ESS=np.zeros(dim)
#         for d in range(dim): ESS[d]=ess(samp[:,d])
        # ESS=np.array([ess(col) for col in samp.T])
        ESS=Parallel(n_jobs=4)(map(delayed(ess), np.transpose(samp)))
        return num_samp,ESS

    def plot_samp(self,samp,loglik,alg_no):
        num_samp,dim=np.shape(samp)
        idx=np.floor(np.linspace(0,num_samp-1,np.min([1e4,num_samp]))).astype(int)
#         col=np.sort(np.random.choice(dim,np.min([4,dim]),False))
#         col=np.array([1,2,np.floor(dim/2),dim],dtype=np.int)-1
        col=np.arange(6,dtype=np.int)
#         mat4plot=samp[idx,]; mat4plot=mat4plot[:,col]; # samp[idx,col] doesn't work, seems very stupid~~
        mat4plot=samp[np.ix_(idx,col)]
        # figure 1: plot selected samples
        fig,axes = plt.subplots(nrows=1,ncols=2,num=alg_no*2,figsize=(10,6))
        [axes[0].plot(idx,samp[idx,d]) for d in col]
        axes[0].set_title('selected samples')
        axes[1].plot(loglik)
        axes[1].set_title('log-likelihood')
        if self.save_fig:
            fig.savefig(os.path.join(self.savepath,self.algs[alg_no]+'_traceplot.png'),dpi=fig.dpi)
        else:
            plt.show()
        # figure 2: pairwise distribution density contour
        from scipy import stats
        def corrfunc(x, y, **kws):
            r, _ = stats.pearsonr(x, y)
            ax = plt.gca()
            ax.annotate("r = {:.2f}".format(r),
                        xy=(.1, .9), xycoords=ax.transAxes)

#         fig = plt.figure(num=alg_no+self.num_algs,figsize=(8,8))
        df4plot = pd.DataFrame(mat4plot,columns=[r'$\theta_{%d}$' % k for k in col])
#         pd.scatter_matrix(df4plot)
#         plt.figure(alg_no+self.num_algs)
        g  = sns.PairGrid(df4plot)
        g.map_upper(plt.scatter)
        g.map_lower(sns.kdeplot, cmap="Blues_d")
        g.map_diag(sns.kdeplot, lw=3, legend=False)
        g.map_lower(corrfunc)
#         if matplotlib.get_backend().lower() in ['agg', 'macosx']:
#             fig.set_tight_layout(True)
#         else:
#             fig.tight_layout()
        if self.save_fig:
            g.savefig(os.path.join(self.savepath,self.algs[alg_no]+'_distribution.png'))
        else:
            plt.show()

    def analyze(self):
        self.acptrate=np.zeros(self.num_algs)
        self.spiter=np.zeros(self.num_algs)
        self.ESS=np.zeros((self.num_algs,4))
        self.minESS_s=np.zeros(self.num_algs)
        self.spdup=np.zeros(self.num_algs)
        self.PDEsolns=np.zeros(self.num_algs)

        for a in range(self.num_algs):
            acpt_=[];time_=[];ESS_=[];solns_=[]; found_a=False
            for f_i in self.fnames:
                if '_'+self.algs[a]+'_' in f_i:
                    f=open(os.path.join(self.savepath,f_i),'rb')
                    _,_,_,samp,loglik,acpt,time,_=pickle.load(f,encoding='bytes') # Unpickling a python 2 object with python 3
                    _,_,_,_,_,_,_,soln_count,_=pickle.load(f,encoding='bytes')
#                     soln_count=[0]
                    f.close()
                    num_samp,ESS_q_i=self.cal_ESS(samp)
                    _,ESS_l_i=self.cal_ESS(loglik[:,np.newaxis])
                    acpt_.append(acpt);time_.append(time)
                    ESS_.append([np.min(ESS_q_i),np.median(ESS_q_i),np.max(ESS_q_i),ESS_l_i[0]])
                    solns_.append(sum(soln_count))
                    found_a=True

            if found_a:
                self.acptrate[a]=np.mean(acpt_)
                self.spiter[a]=np.mean(time_)/num_samp
                self.ESS[a,]=np.mean(ESS_,axis=0)
                self.minESS_s[a]=self.ESS[a,0]/np.mean(time_)
                print('Efficiency measurement (min,med,max,ll,minESS/s) for %s algorithm is: ' % self.algs[a])
                print([ "{:0.5f}".format(x) for x in np.append(self.ESS[a,],self.minESS_s[a])])
                self.PDEsolns[a]=np.mean(solns_)

                if self.PLOT:
                    self.plot_samp(samp, loglik, a)
        # speed up
        self.spdup=self.minESS_s/self.minESS_s[0]

        # summary table
        ESS_str=[np.array2string(ess_a,precision=2,separator=',').replace('[','').replace(']','') for ess_a in self.ESS]
        self.sumry_np=np.array([self.algs,self.acptrate,self.spiter,ESS_str,self.minESS_s,self.spdup,self.PDEsolns]).T
        sumry_header=('Method','AR','s/iter','ESS (min,med,max)','minESS/s','spdup','PDEsolns')
        self.sumry_pd=pd.DataFrame(data=self.sumry_np,columns=sumry_header)

        # save it to text
        if self.save_txt:
            np.savetxt(os.path.join(self.savepath,'summary.txt'),self.sumry_np,fmt="%s",delimiter=',',header=','.join(sumry_header))
            self.sumry_pd.to_csv(os.path.join(self.savepath,'summary.csv'),index=False,cols=sumry_header)

        return object

if __name__ == '__main__':
    algs=('pCN','infMALA','infHMC','infmMALA','infmHMC','splitinfmMALA','splitinfmHMC')
#     dim=100
#     algs=[s+'_dim'+str(dim) for s in algs]
    print('Analyzing posterior samples ...\n')
    _=ana_samp(algs=algs,dir_name='analysis',PLOT=True,save_fig=True).analyze()