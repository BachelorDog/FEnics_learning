"""
QR decomposition in weighted space.
Input:
Y: m x n target matrix;
W: m x m pd matrix, weight matrix;
Output:
Q: m x n, R: n x n such that
Y=QR and Q'WQ=I
----------------------------------
Shiwei Lan @ Caltech, 2017
----------------------------------
[1]Lowery BR, Langou J. 
Stability analysis of QR factorization in an oblique inner product. 
arXiv preprint arXiv:1401.5171, 2014
[2]Arvind K. Saibaba, Jonghyun Lee, Peter K. Kitanidis,
Randomized algorithms for Generalized Hermitian Eigenvalue Problems with application to computing Karhunen-Loeve expansion,
Numerical Linear Algebra with Applications 23 (2), pp. 314-339.
"""

import numpy as np

def CholQR(Y,W):
    """
    CholQR with W-inner products
    """
    Z=W.dot(Y) if type(W) is np.ndarray else np.array([W(r) for r in Y.T]).T
    C=Y.T.dot(Z)
    L=np.linalg.cholesky(C)
    Q=np.linalg.solve(L,Y.T).T
#     WQ=np.linalg.solve(L,Z.T).T
#     return Q,WQ,L.T
    return Q,L.T

def preCholQR(Y,W):
    """
    Pre-CholQR with W-inner products
    """
    Z,S=np.linalg.qr(Y)
#     Q,WQ,U=CholQR(Z, W)
    Q,U=CholQR(Z, W)
    R=U.dot(S)
#     return Q,WQ,R
    return Q,R