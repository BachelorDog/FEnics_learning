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

import sys
sys.path.append( "../" )
from util.bayesianStats import effectiveSampleSize as ess
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
        if dim==1:
            ESS=ess(samp)
        else:
            ESS=Parallel(n_jobs=4)(map(delayed(ess), np.transpose(samp)))
        return num_samp,ESS

    def plot_samp(self,samp,loglik,alg_no):
        coord=np.int_(samp[0,]); samp=samp[1:,]
        num_samp,dim=np.shape(samp)
        idx=np.floor(np.linspace(0,num_samp-1,np.min([1e4,num_samp]))).astype(int)
#         col=np.sort(np.random.choice(dim,np.min([4,dim]),False))
#         col=np.array([1,2,np.floor(dim/2),dim],dtype=np.int)-1
        col=np.arange(np.min([6,dim]),dtype=np.int)
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
        self.stepsize=np.zeros(self.num_algs)
        self.acptrate=np.zeros(self.num_algs)
        self.spiter=np.zeros(self.num_algs)
        self.ESS=np.zeros((self.num_algs,4))
        self.minESS_s=np.zeros(self.num_algs)
        self.spdup=np.zeros(self.num_algs)
        self.PDEsolns=np.zeros(self.num_algs)
        
        # calculate ESS of samples stored in h5 file separately
        ESS_fname=os.path.join(self.savepath,'sumry_ESS.txt')
        if not os.path.isfile(ESS_fname):
            import subprocess
            subprocess.call('./get_ESS.sh')
        else:
#             sumry_ESS=np.array(np.loadtxt(ESS_fname,dtype={'names':('method','ESS_min','ESS_med','ESS_max'),'formats':('|S10',np.float,np.float,np.float)},delimiter=','))[None,]
            sumry_ESS=np.array(np.genfromtxt(ESS_fname,dtype=np.dtype([('method','U11'),('ESS_min','<f8'),('ESS_med','<f8'),('ESS_max','<f8')]),delimiter=','),ndmin=1)
        
        for a in range(self.num_algs):
            # samples' ESS
            _stepsz=[];_acpt=[];_time=[];_ESS=[];_ESS_l=[];_solns=[]
            if self.algs[a] in sumry_ESS[:]['method']:
                _a=sumry_ESS[:]['method'].tolist().index(self.algs[a])
                self.ESS[a,:3]=[sumry_ESS[_a][k] for k in range(1,4)]
            # other quantities
            found_a=False
            for f_i in self.fnames:
                if '_'+self.algs[a]+'_' in f_i:
                    try:
                        f=open(os.path.join(self.savepath,f_i),'rb')
#                         _,_,_,loglik,acpt,time=pickle.load(f,encoding='bytes') # Unpickling a python 2 object with python 3
#                         _,_,_,_,_,soln_count,args=pickle.load(f,encoding='bytes')
                        f_read=pickle.load(f,encoding='bytes')
                        stepsz=f_read[0 if any([s in self.algs[a] for s in ['DILI','aDRinf']]) else 1] # todo: DILI has two step sizes
#                         stepsz=f_read[0]
#                         adjuster=-(self.algs[a]=='DILI' or self.algs[a]=='aDRinfmMALA')+('aDRinfmHMC' in self.algs[a])
                        adjuster=-(self.algs[a]=='DILI')+('aDRinf' in self.algs[a])
                        loglik,acpt,time=f_read[adjuster+3:adjuster+6]
                        f_read=pickle.load(f,encoding='bytes')
                        soln_count,args=f_read[5:7]
                        f.close()
                        _,_ESS_l_i=self.cal_ESS(loglik[:,np.newaxis])
                        _stepsz.append(stepsz);_acpt.append(acpt);_time.append(time)
                        _ESS_l.append(_ESS_l_i[0])
                        _solns.append(sum(soln_count))
                        found_a=True
                    except:
                        pass
            if found_a:
                self.stepsize[a]=np.mean(_stepsz)
                self.acptrate[a]=np.mean(_acpt)
                num_samp=int(str(args).split("'num_samp'=")[-1].split(",")[0])
                self.spiter[a]=np.mean(_time)/num_samp
                self.ESS[a,3]=np.mean(_ESS_l)
                self.minESS_s[a]=self.ESS[a,0]/np.mean(_time)
                print('Efficiency measurement (min,med,max,ll,minESS/s) for %s algorithm is: ' % self.algs[a])
                print([ "{:0.5f}".format(x) for x in np.append(self.ESS[a,],self.minESS_s[a])])
                self.PDEsolns[a]=np.mean(_solns)

                if self.PLOT:
                    # select samples for plot
                    samp4plot_fname=os.path.join(self.savepath,self.algs[a]+'_selected_samples.txt')
                    if os.path.isfile(samp4plot_fname):
                        samp4plot=np.loadtxt(samp4plot_fname,delimiter=',')
                    else:
                        import subprocess
                        subprocess.call('./get_ESS.sh')
                    self.plot_samp(samp4plot, loglik, a)
        # speed up
        self.spdup=self.minESS_s/self.minESS_s[0]

        # summary table
        ESS_str=[np.array2string(ess_a,precision=2,separator=',').replace('[','').replace(']','') for ess_a in self.ESS]
        self.sumry_np=np.array([self.algs,self.stepsize,self.acptrate,self.spiter,ESS_str,self.minESS_s,self.spdup,self.PDEsolns]).T
        sumry_header=('Method','h','AR','s/iter','ESS (min,med,max)','minESS/s','spdup','PDEsolns')
        self.sumry_pd=pd.DataFrame(data=self.sumry_np,columns=sumry_header)

        # save it to text
        if self.save_txt:
            np.savetxt(os.path.join(self.savepath,'summary.txt'),self.sumry_np,fmt="%s",delimiter=',',header=','.join(sumry_header))
            self.sumry_pd.to_csv(os.path.join(self.savepath,'summary.csv'),index=False,header=sumry_header)

        return object

if __name__ == '__main__':
    algs=('pCN','infMALA','infHMC','DRinfmMALA','DRinfmHMC','DILI','aDRinfmMALA','aDRinfmHMC')
    print('Analyzing posterior samples ...\n')
    SNR=100
    _=ana_samp(algs=algs,dir_name='analysis_f_SNR'+str(SNR),PLOT=False,save_fig=True).analyze()