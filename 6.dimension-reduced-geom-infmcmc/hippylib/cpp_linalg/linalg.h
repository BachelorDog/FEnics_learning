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

#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/common/Array.h>

namespace dolfin
{

class cpp_linalg
{
public:
	//out = A*B
	std::shared_ptr<Matrix> MatMatMult(const GenericMatrix & A, const GenericMatrix & B);
	//out = Pt*A*P
	std::shared_ptr<Matrix> MatPtAP(const GenericMatrix & A, const GenericMatrix & P);
	//out = At*B
	std::shared_ptr<Matrix> MatAtB(const GenericMatrix & A, const GenericMatrix & B);
	//out = At
	std::shared_ptr<Matrix> Transpose(const GenericMatrix & A);

	void SetToOwnedGid(GenericVector & v, std::size_t gid, double val);
	double GetFromOwnedGid(const GenericVector & v, std::size_t gid);
};

class MultiVector
{
public:
	MultiVector();
	MultiVector(const GenericVector & v, int nvec);
	MultiVector(const MultiVector & orig);

	int nvec() const {return mv.size();}

	void setSizeFromVector(const GenericVector & v, int nvec);

	std::shared_ptr<const GenericVector> operator[](int i) const;
	std::shared_ptr<GenericVector> operator[](int i);

	std::shared_ptr<const GenericVector> __getitem__(int i) const
	{
		return mv[i];
	}

	std::shared_ptr<GenericVector> __setitem__(int i)
	{
		return mv[i];
	}

	// m[i] = this[i] \cdot v
	void dot(const GenericVector & v, Array<double> & m);

	// m[i,j] = this[i] \cdot other[j]
	void dot(const MultiVector & other, Array<double> & m);

	// v += sum_i alpha[i]*this[i]
	void reduce(GenericVector & v, const Array<double> & alpha);

	void axpy(double a, const GenericVector & y);
	void axpy(const Array<double> & a, const MultiVector & y);

	// this[k] *= a
	void scale(int k, double a);

	// this[k] *= a[k]
	void scale(const Array<double> & a);

	void zero();

	void norm_all(const std::string norm_type, Array<double> & norms);

	void swap(MultiVector & other);

	~MultiVector();

private:
	void dot_self(Array<double> & m);

	std::vector<std::shared_ptr<GenericVector> > mv;
};

}
