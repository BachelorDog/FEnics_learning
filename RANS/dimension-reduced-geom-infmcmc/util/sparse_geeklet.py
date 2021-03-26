#!/usr/bin/env python
"""
Some handy functions to efficiently manipulate scipy sparse matrix (e.g. in csr format)
Shiwei Lan @ U of Warwick, 2016; @ Caltech, Sept. 2016
"""

import numpy as np
import scipy.sparse as sps

# Convert PETScMatrix to csr_matrix
# petsc4py must be compiled with dolfin!
try:
    import dolfin as df
    petscmat2csr = lambda matrix: sps.csr_matrix(tuple(df.as_backend_type(matrix).mat().getValuesCSR()[::-1]),
                                                 shape=(matrix.size(0),matrix.size(1)))
except ImportError:
    print('Dolfin not installed!')
    pass

# Convert csr_matrix to pestc_mat
# petsc4py must be compiled with dolfin!
try:
    from petsc4py import PETSc
    csr2petscmat = lambda matrix: PETSc.Mat().createAIJ(size=matrix.shape,csr=(matrix.indptr,matrix.indices,matrix.data))
except ImportError:
    print('petsc4py not installed!')
    pass

## some functions to efficiently zero-out certain rows in csr matrix ##
## http://stackoverflow.com/questions/19784868/what-is-most-efficient-way-of-setting-row-to-zeros-for-a-sparse-scipy-matrix ##

def csr_row_set_nz_to_val(csr, row, value=0):
    """Set all nonzero elements (elements currently in the sparsity pattern)
    to the given value. Useful to set to 0 mostly.
    """
    if not isinstance(csr, sps.csr_matrix):
        raise ValueError('Matrix given must be of CSR format.')
    csr.data[csr.indptr[row]:csr.indptr[row+1]] = value

def csr_rows_set_nz_to_val(csr, rows, value=0):
    for row in rows:
        csr_row_set_nz_to_val(csr, row)
    if value == 0:
        csr.eliminate_zeros()

def csr_zero_rows(csr, rows_to_zero): # i like this one better
    rows, cols = csr.shape
    mask = np.ones((rows,), dtype=np.bool)
    mask[rows_to_zero] = False
    nnz_per_row = np.diff(csr.indptr)

    mask = np.repeat(mask, nnz_per_row)
    nnz_per_row[rows_to_zero] = 0
    csr.data = csr.data[mask]
    csr.indices = csr.indices[mask]
    csr.indptr[1:] = np.cumsum(nnz_per_row)

## ------------------end--of--functions---------------- ##

## keep certain rows in csr matrix ##

def csr_keep_rows(csr, rows_to_keep): # modified from above
    rows, cols = csr.shape
    labl = np.zeros((rows,), dtype=np.bool)
    labl[rows_to_keep] = True
    nnz_per_row = np.diff(csr.indptr)

    labl = np.repeat(labl, nnz_per_row)
    nnz_per_row_kept = np.zeros_like(nnz_per_row)
    nnz_per_row_kept[rows_to_keep] = nnz_per_row[rows_to_keep]
    csr.data = csr.data[labl]
    csr.indices = csr.indices[labl]
    csr.indptr[1:] = np.cumsum(nnz_per_row_kept)

# Trim small entries of csr_matrix to zero by some threshold
def csr_trim0(mat_sps,threshold=1e-10):
    mat_sps.data *= abs(mat_sps.data)>=threshold
    mat_sps.eliminate_zeros()
#     return mat_sps


## Save / load scipy sparse csr_matrix in portable data format ##
# http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format ##

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sps.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])
