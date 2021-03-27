#include <dolfin/la/PETScVector.h>
#include "PRNG.h"

using namespace dolfin;

Random::Random(int seed):
		eng(seed),
		d_normal(0.,1.),
		d_uniform(0., 1.)
{

}

void Random::seed(int seed)
{
	eng.seed(seed);
}

void Random::split(int _rank, int _nproc, int _block_size, int seed)
{
	eng.seed(seed);
	eng.split(_rank, _nproc, _block_size);
}

double Random::uniform(double a, double b)
{
	double r = d_uniform( eng );
	return a + (b-a)*r;
}

double Random::normal(double mu, double sigma)
{
	double z = d_normal( eng );
	return mu + sigma*z;
}

double Random::rademacher()
{
	bool val = d_bernoulli( eng );
	if(val)
		return 1.;
	else
		return -1.;
}

void Random::uniform(GenericVector & v, double a, double b)
{
	PETScVector* vec = &as_type<PETScVector>(v);
	Vec vv = vec->vec();

	PetscInt local_size;
	VecGetLocalSize(vv, &local_size);

	PetscScalar *data = NULL;
	VecGetArray(vv, &data);

	for(PetscInt i = 0; i < local_size; ++i)
		data[i] = a + (b-a)*d_uniform( eng );

	VecRestoreArray(vv, &data);
}

void Random::normal(GenericVector & v, double sigma, bool zero_out)
{
	if(zero_out)
		v.zero();

	PETScVector* vec = &as_type<PETScVector>(v);
	Vec vv = vec->vec();

	PetscInt local_size;
	VecGetLocalSize(vv, &local_size);

	PetscScalar *data = NULL;
	VecGetArray(vv, &data);

	for(PetscInt i = 0; i < local_size; ++i)
		data[i] += sigma*d_normal( eng );

	VecRestoreArray(vv, &data);
}

void Random::rademacher(GenericVector & v)
{
	PETScVector* vec = &as_type<PETScVector>(v);
	Vec vv = vec->vec();

	PetscInt local_size;
	VecGetLocalSize(vv, &local_size);

	PetscScalar *data = NULL;
	VecGetArray(vv, &data);

	for(PetscInt i = 0; i < local_size; ++i)
	{
		auto val = d_bernoulli( eng );
		if(val)
			data[i] = 1.;
		else
			data[i] = -1.;
	}

	VecRestoreArray(vv, &data);
}

