"""
A class of methods for calculating (partial) eigen-problems.
Shiwei Lan @ Caltech, 2016
"""

import os
import dolfin as df
import numpy as np

def eigen_xct(V,A,B=None,k=100,spect='LM',**kwargs):
    """
    Get eigen-pairs of A (or generalized eigen-pairs for (A,B)) for the first k in the spectrum spect.
    """
    if 'mpi_comm' in kwargs:
        mpi_comm=kwargs['mpi_comm']
    else:
        mpi_comm=df.mpi_comm_world()
    # mixed functions to store eigenfunctions
    try:
        M=df.FunctionSpace(V.mesh(),df.MixedElement([V.ufl_element()]*k))
    except:
        print('Warning: ''MixedFunctionSpace'' has been deprecated after DOLFIN version 1.6.0.')
        M=df.MixedFunctionSpace([V]*k)
    # Extract subfunction dofs
    eigf_dofs = [M.sub(i).dofmap().dofs() for i in range(k)]
    eigf=df.Function(M)
    # try reading eigen-pairs from file
    eig_files=[f for f in os.listdir(os.getcwd()) if f.startswith('prior_'+spect+'eig')]
    found_eigv=False; found_eigf=False
    if any(eig_files):
        for f_i in eig_files:
            if int(f_i.split("_k")[-1].split(".")[0])>=k:
                if f_i.endswith('.txt'):
                    try:
                        eigv_=np.loadtxt(os.path.join(os.getcwd(),f_i),delimiter=',')
                        eigv=eigv_[:k]
                        found_eigv=True
                    except:
                        pass
                if f_i.endswith('.h5'):
                    try:
                        f=df.HDF5File(mpi_comm,os.path.join(os.getcwd(),f_i),"r")
                        eigf_i=df.Function(V,name='eigenfunction')
                        for i,dof_i in enumerate(eigf_dofs):
                            f.read(eigf_i,'eigf_{0}'.format(i))
                            eigf.vector()[dof_i]=eigf_i.vector()
                        f.close()
                        found_eigf=True
                    except:
                        f.close()
                        pass
                if found_eigv and found_eigf:
                    break
    if found_eigv and found_eigf:
        print('Read the first %d eigen-pairs with '% k+{'LM':'largest magnitude','SM':'smallest magnitude'}[spect]+' successfully!')
        return (eigv,eigf),eigf_dofs
    else:
        # solve the associated eigen-problem
        f=df.HDF5File(mpi_comm,os.path.join(os.getcwd(),'prior_'+spect+'eigf_k'+str(k)+'.h5'),"w")
        eigf_i=df.Function(V,name='eigenfunction')
        if type(A) is df.PETScMatrix and df.has_slepc():
            # using SLEPc
            print("Computing the first %d eigenvalues with "% k+{'LM':'largest magnitude','SM':'smallest magnitude'}[spect]+"...")
            # Create eigen-solver
            if B is not None and type(B) is df.PETScMatrix:
                eigen = df.SLEPcEigenSolver(A,B)
                eigen.parameters['problem_type']='gen_hermitian'
            else:
                eigen = df.SLEPcEigenSolver(A)
#                 eigen.parameters['problem_type']='hermitian'
            eigen.parameters['spectrum']={'LM':'largest magnitude','SM':'smallest magnitude'}[spect]
            if spect is 'SM':
                eigen.parameters['tolerance']=1e-10
                eigen.parameters['maximum_iterations']=100
                eigen.parameters['spectral_transform']='shift-and-invert'
                eigen.parameters['spectral_shift']=10.0*df.DOLFIN_EPS
            eigen.solve(k)
            print('Total number of iterations: %d, and total number of convergences: %d.' %(eigen.get_iteration_number(),eigen.get_number_converged()))
            # get eigen-pairs
            eigv=np.zeros(k)
            for i,dof_i in enumerate(eigf_dofs):
                eigv[i],_,eigf_vec_i,_=eigen.get_eigenpair(i)
                eigf_i.vector()[:]=eigf_vec_i[:]
                f.write(eigf_i,'eigf_{0}'.format(i))
                eigf.vector()[dof_i]=eigf_i.vector()
        else:
            warnings.warn('petsc4py or SLEPc not found! Using scipy.sparse.linalg.eigsh...')
            import scipy.sparse.linalg as spsla
            eigv, eigf_vec = spsla.eigsh(A,k=k,M=B,which=spect)
            dsc_ord = eigv.argsort()[::-1]
            eigv = eigv[dsc_ord]; eigf_vec = eigf_vec[:,dsc_ord]
            for i,dof_i in enumerate(eigf_dofs):
                eigf_i.vector()[:]=eigf_vec[:,i]
                f.write(eigf_i,'eigf_{0}'.format(i))
                eigf.vector()[dof_i]=eigf_i.vector()
        f.close()
        np.savetxt(os.path.join(os.getcwd(),'prior_'+spect+'eigv_k'+str(k)+'.txt'),eigv,delimiter=',')
        return (eigv,eigf),eigf_dofs

def _eigen_randproj(Omega,Y):
    """
    Eigen-decomposition (1pass) of A based on random projection Y=A *Omega
    key component of randomized eigen-decomposition algorithms.
    """
    Q,_=np.linalg.qr(Y)
    QOmega_t=Omega.T.dot(Q)
    QY_t=Y.T.dot(Q)
    B_appx_t=np.linalg.solve(QOmega_t,QY_t)
    B_appx=.5*(B_appx_t+B_appx_t.T)
    eigv,V=np.linalg.eigh(B_appx)
    dsc_ord=eigv.argsort()[::-1]
    eigv=eigv[dsc_ord]
    V=V[:,dsc_ord]
    eigf_vec=Q.dot(V)
    eigs = eigv,eigf_vec
    
    return eigs

def eigen_RA_rank(A,dim=None,k=20,p=10):
    """
    Get partial eigen-pairs of linear operator A based on the threshold using randomized algorithms for fixed rank.
    Nathan Halko, Per Gunnar Martinsson, and Joel A. Tropp,
    Finding structure with randomness:
    Probabilistic algorithms for constructing approximate matrix decompositions,
    SIAM Review, 53 (2011), pp. 217-288.
    --credit to: Umberto Villa
    """
    if dim is None:
        dim=A.shape[0]
    Omega=np.random.randn(dim,k+p)
    Y=A.dot(Omega) if type(A) is np.ndarray else np.array([A(r) for r in Omega.T]).T
    eigv,eigf_vec=_eigen_randproj(Omega,Y)
    eigv=eigv[:k]
    eigf_vec=eigf_vec[:,:k]
    eigs = eigv,eigf_vec
    
    return eigs

def eigen_RA_prec(A,dim=None,threshold=0.01,increment_k=20,p=10):
    """
    Get partial eigen-pairs of linear operator A based on the threshold using randomized algorithms for fixed precision.
    Nathan Halko, Per Gunnar Martinsson, and Joel A. Tropp,
    Finding structure with randomness:
    Probabilistic algorithms for constructing approximate matrix decompositions,
    SIAM Review, 53 (2011), pp. 217-288.
    """
    if dim is None:
        dim=A.shape[0]
    Omega=np.random.randn(dim,increment_k+p)
    Y=A.dot(Omega) if type(A) is np.ndarray else np.array([A(r) for r in Omega.T]).T
    eigv=np.zeros(0); eigf_vec=np.zeros((dim,0))
    num_eigs=0
    while num_eigs<dim+np.float_(increment_k)/2:
        eigv_k,eigf_vec_k=_eigen_randproj(Omega,Y)
        eigv_k=eigv_k[:increment_k]
        eigf_vec_k=eigf_vec_k[:,:increment_k]
        # threshold
        idx = eigv_k>=threshold
        eigv=np.append(eigv,eigv_k[idx])
        eigf_vec=np.append(eigf_vec,eigf_vec_k[:,idx],axis=1)
        if sum(idx)<increment_k:
            break
        else:
            Y-=(eigf_vec_k*eigv_k).dot(eigf_vec_k.T).dot(Omega)
        num_eigs+=increment_k
    eigs = eigv,eigf_vec
    
    return eigs

def eigen_RA(A,dim=None,**kwargs):
    """
    Get partial eigen-pairs of linear operator A.
    """
    if 'k' in kwargs:
        eigs = eigen_RA_rank(A,dim,**kwargs)
    elif 'threshold' in kwargs:
        eigs = eigen_RA_prec(A,dim,**kwargs)
    else:
        print('warning: algorithm not found!')
        pass
    return eigs

def _geigen_randproj(Omega,Y_bar,Y,B):
    """
    Generalized eigen-decomposition (1pass) of (A,B) based on random projections Y_bar=A Omega and Y=invB Y_bar
    key component of randomized generalized eigen-decomposition algorithms.
    """
    #-------- begin pre-CholQR(Y,B) -------#
    Z,_=np.linalg.qr(Y)
    BZ=B.dot(Z) if type(B) is np.ndarray else np.array([B(r) for r in Z.T]).T
    R=np.linalg.cholesky(Z.T.dot(BZ))
    Q=np.linalg.solve(R,Z.T).T
    #-------- end pre-CholQR(Y,B) -------#
    BQ=np.linalg.solve(R,BZ.T).T
    Xt=Omega.T.dot(BQ)
    Wt=Y_bar.T.dot(Q)
    Tt=np.linalg.solve(Xt,Wt)
    T=.5*(Tt+Tt.T)
    eigv,V=np.linalg.eigh(T)
    dsc_ord=eigv.argsort()[::-1]
    eigv=eigv[dsc_ord]
    V=V[:,dsc_ord]
    eigf_vec=Q.dot(V)
    eigs = eigv,eigf_vec
    
    return eigs

def geigen_RA_rank(A,B,invB,dim=None,k=20,p=10):
    """
    Get partial generalized eigen-pairs of pencile (A,B) based on the threshold using randomized algorithms for fixed rank.
    Arvind K. Saibaba, Jonghyun Lee, Peter K. Kitanidis,
    Randomized algorithms for Generalized Hermitian Eigenvalue Problems with application to computing Karhunen-Loeve expansion,
    Numerical Linear Algebra with Applications 23 (2), pp. 314-339.
    --credit to: Umberto Villa
    """
    if dim is None:
        dim=A.shape[0]
    Omega=np.random.randn(dim,k+p)
    if all([type(x) is np.ndarray for x in (A,invB)]):
        Y_bar=A.dot(Omega)
        Y=invB.dot(Y_bar)
    else:
        Y_bar=np.zeros((dim,k+p))
        Y=np.zeros((dim,k+p))
        for i in range(k+p):
            Y_bar[:,i]=A(Omega[:,i])
            Y[:,i]=invB(Y_bar[:,i])
    eigv,eigf_vec=_geigen_randproj(Omega,Y_bar,Y,B)
    eigv=eigv[:k]
    eigf_vec=eigf_vec[:,:k]
    eigs = eigv,eigf_vec
    
    return eigs

def geigen_RA_prec(A,B,invB,dim=None,threshold=0.01,increment_k=20,p=10):
    """
    Get partial generalized eigen-pairs of pencile (A,B) based on the threshold using randomized algorithms for fixed precision.
    Arvind K. Saibaba, Jonghyun Lee, Peter K. Kitanidis,
    Randomized algorithms for Generalized Hermitian Eigenvalue Problems with application to computing Karhunen-Loeve expansion,
    Numerical Linear Algebra with Applications 23 (2), pp. 314-339.
    --credit to: Umberto Villa
    """
    if dim is None:
        dim=A.shape[0]
    Omega=np.random.randn(dim,increment_k+p)
    if all([type(x) is np.ndarray for x in (A,invB)]):
        Y_bar=A.dot(Omega)
        Y=invB.dot(Y_bar)
    else:
        Y_bar=np.zeros((dim,increment_k+p))
        Y=np.zeros((dim,increment_k+p))
        for i in range(increment_k+p):
            Y_bar[:,i]=A(Omega[:,i])
            Y[:,i]=invB(Y_bar[:,i])
    eigv=np.zeros(0); eigf_vec=np.zeros((dim,0))
    num_eigs=0
    BOmega=None
    while num_eigs<dim+np.float_(increment_k)/2:
        eigv_k,eigf_vec_k=_geigen_randproj(Omega,Y_bar,Y,B)
        eigv_k=eigv_k[:increment_k]
        eigf_vec_k=eigf_vec_k[:,:increment_k]
        # threshold
        idx = eigv_k>=threshold
        eigv=np.append(eigv,eigv_k[idx])
        eigf_vec=np.append(eigf_vec,eigf_vec_k[:,idx],axis=1)
        if sum(idx)<increment_k:
            break
        else:
            if BOmega is None:
                BOmega=B.dot(Omega) if type(B) is np.ndarray else np.array([B(r) for r in Omega.T]).T
            Y-=(eigf_vec_k*eigv_k).dot(eigf_vec_k.T).dot(BOmega)
            Y_bar=np.array([B(Y[:,i]) for i in range(increment_k+p)]).T
        num_eigs+=increment_k
    eigs = eigv,eigf_vec
    
    return eigs

def geigen_RA(A,B,invB,dim=None,**kwargs):
    """
    Get partial generalized eigen-pairs of pencile (A,B).
    """
    if 'k' in kwargs:
        eigs = geigen_RA_rank(A,B,invB,dim,**kwargs)
    elif 'threshold' in kwargs:
        eigs = geigen_RA_prec(A,B,invB,dim,**kwargs)
    else:
        print('warning: algorithm not found!')
        pass
    return eigs

if __name__ == '__main__':
    np.random.seed(2016)
    N=50
    I=np.eye(N)
    P=np.random.permutation(np.arange(N))
    O=I[P]
    # eigen-problem
    print('\nTesting eigenvalue problem...')
    d=10*np.exp(-np.sort(np.arange(N)+np.random.uniform(-.5,.5,N))/2)
    print('True eigen-values:')
    print(d)
#     print(O)
    A=O.dot(np.diag(d)).dot(O.T)
    k=10
    eigs=eigen_RA_rank(A, k=k)
    print('Fixed %d rank solution:' % k)
    print(eigs[0])
#     print(eigs[1])
    prec=1e-3
    eigs=eigen_RA_prec(A, increment_k=10,threshold=prec)
    print('Fixed precision %e solution:' % prec)
    print(eigs[0])
    
    # generalized eigen-problem
    print('\nTesting generalized eigenvalue problem...')
    d1=10*np.exp(-np.sort(np.arange(N)+np.random.uniform(-.5,.5,N))/4)
    B=O.dot(np.diag(d1)).dot(O.T)
    import scipy.linalg as spla
    geigs_AB=spla.eigh(A,b=B)
    dsc_ord=geigs_AB[0].argsort()[::-1]
    print('True (generalized) eigen-values:')
    print(geigs_AB[0][dsc_ord])
    invB=O.dot(np.diag(1.0/d1)).dot(O.T)
    k=10
    eigs=geigen_RA_rank(A,B,invB, k=k,p=10)
    print('Fixed %d rank solution:' % k)
    print(eigs[0])
#     print(eigs[1])
    prec=1e-2
    eigs=geigen_RA_prec(A,B,invB, threshold=prec,p=10)
    print('Fixed precision %e solution:' % prec)
    print(eigs[0])