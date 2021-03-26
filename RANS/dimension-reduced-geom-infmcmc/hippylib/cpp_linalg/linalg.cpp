/* Copyright (c) 2016, The University of Texas at Austin & University of
 * California, Merced.
 *
 * All Rights reserved.
 * See file COPYRIGHT for details.
 *
 * This file is part of the hIPPYlib library. For more information and source
 * code availability see https://hippylib.github.io.
 *
 * hIPPYlib is free software; you can redistribute it and/or modify it under the
 * terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 3.0 dated June 2007.
*/

#include "linalg.h"
#include <dolfin/la/PETScVector.h>
#include <dolfin/la/GenericLinearSolver.h>
#include <dolfin/common/Timer.h>

#include <cassert>

namespace dolfin
{

std::shared_ptr<Matrix> cpp_linalg::MatMatMult(const GenericMatrix & A, const GenericMatrix & B)
{
    const PETScMatrix* Ap = &as_type<const PETScMatrix>(A);
    const PETScMatrix* Bp = &as_type<const PETScMatrix>(B);
    Mat CC;
    PetscErrorCode ierr = ::MatMatMult(Ap->mat(), Bp->mat(), MAT_INITIAL_MATRIX, PETSC_DEFAULT, &CC);

    ISLocalToGlobalMapping rmappingA;
    ISLocalToGlobalMapping cmappingB;
    MatGetLocalToGlobalMapping(Ap->mat(),&rmappingA,NULL);
    MatGetLocalToGlobalMapping(Bp->mat(),NULL, &cmappingB);

    MatSetLocalToGlobalMapping(CC, rmappingA, cmappingB);

    PETScMatrix CCC = PETScMatrix(CC);
    MatDestroy(&CC);
    return std::shared_ptr<Matrix>( new Matrix(CCC) );
}

std::shared_ptr<Matrix> cpp_linalg::MatPtAP(const GenericMatrix & A, const GenericMatrix & P)
{
	const PETScMatrix* Ap = &as_type<const PETScMatrix>(A);
	const PETScMatrix* Pp = &as_type<const PETScMatrix>(P);
    Mat CC;
    PetscErrorCode ierr = ::MatPtAP(Ap->mat(),Pp->mat(),MAT_INITIAL_MATRIX, 1.0,&CC);

    //Manually set the LocalToGlobalMapping
    ISLocalToGlobalMapping mapping;
    MatGetLocalToGlobalMapping(Pp->mat(),NULL, &mapping);
    MatSetLocalToGlobalMapping(CC, mapping, mapping);


    PETScMatrix CCC = PETScMatrix(CC);
    MatDestroy(&CC);
    return std::shared_ptr<Matrix>( new Matrix(CCC) );
}

std::shared_ptr<Matrix> cpp_linalg::MatAtB(const GenericMatrix & A, const GenericMatrix & B)
{
    const PETScMatrix* Ap = &as_type<const PETScMatrix>(A);
    const PETScMatrix* Bp = &as_type<const PETScMatrix>(B);
    Mat CC;
    PetscErrorCode ierr = MatTransposeMatMult(Ap->mat(), Bp->mat(), MAT_INITIAL_MATRIX, PETSC_DEFAULT, &CC);

    ISLocalToGlobalMapping cmappingA;
    ISLocalToGlobalMapping cmappingB;
    MatGetLocalToGlobalMapping(Ap->mat(),NULL, &cmappingA);
    MatGetLocalToGlobalMapping(Bp->mat(),NULL, &cmappingB);

    MatSetLocalToGlobalMapping(CC, cmappingA, cmappingB);

    PETScMatrix CCC = PETScMatrix(CC);
    MatDestroy(&CC);
    return std::shared_ptr<Matrix>(new Matrix(CCC) );
}

std::shared_ptr<Matrix> cpp_linalg::Transpose(const GenericMatrix & A)
{
	const PETScMatrix* Ap = &as_type<const PETScMatrix>(A);
	Mat At;
	MatTranspose(Ap->mat(), MAT_INITIAL_MATRIX, &At);

	ISLocalToGlobalMapping rmappingA;
	ISLocalToGlobalMapping cmappingA;
	MatGetLocalToGlobalMapping(Ap->mat(),&rmappingA, &cmappingA);
	MatSetLocalToGlobalMapping(At, cmappingA, rmappingA);

	std::shared_ptr<Matrix> out(new Matrix(PETScMatrix(At)));
	MatDestroy(&At);
	return out;
}

void cpp_linalg::SetToOwnedGid(GenericVector & v, std::size_t gid, double val)
{
	assert(v.owns_index(gid));
	la_index index = static_cast<la_index>(gid);
	v.set(&val, 1, &index);
}

double cpp_linalg::GetFromOwnedGid(const GenericVector & v, std::size_t gid)
{
	assert(v.owns_index(gid));
	double val;
	la_index index = static_cast<la_index>(gid);
	v.get(&val, 1, &index);
}

MultiVector::MultiVector()
{
}

MultiVector::MultiVector(const GenericVector & v, int nvec):
		mv(nvec)
{
	for(auto&& vj : mv)
	{
		vj = v.copy();
		vj->zero();
	}
}

MultiVector::MultiVector(const MultiVector & orig):
		mv(orig.mv.size())
{
	int n = mv.size();
	for(int i = 0; i < n; ++i)
		mv[i] = orig.mv[i]->copy();
}

void MultiVector::setSizeFromVector(const GenericVector & v, int nvec)
{
	mv.resize(nvec);
	for(auto&& vj : mv)
	{
		vj = v.copy();
		vj->zero();
	}
}



std::shared_ptr<const GenericVector> MultiVector::operator[](int i) const
{
	return mv[i];
}

std::shared_ptr<GenericVector> MultiVector::operator[](int i)
{
	return mv[i];
}


void MultiVector::dot(const GenericVector & v, Array<double> & m)
{
	double* im = m.data();
	for(auto&& vj : mv)
		*(im++) = vj->inner(v);
}

void MultiVector::dot(const MultiVector & other, Array<double> & m)
{
	if(other.mv.begin() == mv.begin())
		dot_self(m);
	else
	{
		double* data = m.data();
		for(auto&& vi : mv)
			for(auto&& vj : other.mv)
				*(data++) = vi->inner(*vj);
	}
}

void MultiVector::dot_self(Array<double> & m)
{
	int s = mv.size();
	for(int i = 0; i < s; ++i)
	{
		m[i + s*i] = mv[i]->inner(*(mv[i]));
		for(int j = 0; j < i; ++j)
			m[i + s*j] = m[j + s*i] = mv[i]->inner(*(mv[j]));

	}
}

void MultiVector::reduce(GenericVector & v, const Array<double> & alpha)
{
	const double * data = alpha.data();
	for(auto&& vi : mv)
		v.axpy(*(data++), *vi);
}

void MultiVector::axpy(double a, const GenericVector & y)
{
	for(auto&& vi : mv)
		vi->axpy(a, y);
}

void MultiVector::axpy(const Array<double> & a, const MultiVector & y)
{
	int n = nvec();
	assert(a.size() == n);
	assert(y.nvec() == n);

	for(int i = 0; i < n; ++i)
		mv[i]->axpy(a[i], *(y.mv[i]) );
}

void MultiVector::scale(int k, double a)
{
	mv[k]->operator*=(a);
}

void MultiVector::scale(const Array<double> & a)
{
	const double * data = a.data();
	for(auto && vj : mv)
		vj->operator*=(*(data++));
}

void MultiVector::zero()
{
	for(auto&& vi : mv)
		vi->zero();
}

void MultiVector::norm_all(const std::string norm_type, Array<double> & norms)
{
	double * data = norms.data();
	for(auto && vi : mv)
		*(data++) = vi->norm(norm_type);
}

void MultiVector::swap(MultiVector & other)
{
	mv.swap(other.mv);
}

MultiVector::~MultiVector()
{

}



}
