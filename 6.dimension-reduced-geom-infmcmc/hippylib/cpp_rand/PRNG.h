#include <dolfin/la/GenericVector.h>
#include <random>
#include <cassert>

template<class Engine>
class PRNG
{
public:
	typedef typename Engine::result_type result_type;

	explicit PRNG(result_type value = Engine::default_seed):
			eng(value),
			rank(0),
			nproc(0),
			block_size(0),
			is_splitted(false),
			used(0)
	{

	}
	template< class Sseq>
	explicit PRNG( Sseq & s):
		eng(s),
		rank(0),
		nproc(0),
		block_size(0),
		is_splitted(false),
		used(0)
	{

	}

	void seed( result_type value = Engine::default_seed )
	{
		eng.seed(value);
		is_splitted = false;
	}

	template< class Sseq >
	void seed( Sseq& seq )
	{
		eng.seed(seq);
		is_splitted = false;
	}

	result_type operator()()
	{
		if(used < block_size || is_splitted == false)
		{
			++used;
			return eng();
		}
		else
		{
			used = 1;
			eng.discard(nproc*block_size);
			return eng();
		}
	}

	void discard( unsigned long long z )
	{
		assert(is_splitted == false);
		eng.discard(z);
	}

	static constexpr result_type min()
	{
		return Engine::min();
	}

	static constexpr result_type max()
	{
		return Engine::max();
	}

	void split(int _rank, int _nproc, int _block_size)
	{
		if(is_splitted == false)
		{
			eng.discard(_rank*_block_size);
			rank = _rank;
			nproc = _nproc;
			block_size = _block_size;
			is_splitted = true;
		}
		else
		{
			assert( rank == _rank );
			assert( nproc == _nproc );
			assert( block_size == _block_size);
		}
	}

private:
	Engine eng;
	int rank;
	int nproc;
	int block_size;
	bool is_splitted;
	int used;

};

namespace dolfin
{
class Random
{
public:
	Random(int seed);
	void seed(int seed);
	void split(int _rank, int _nproc, int _block_size, int seed);
	double uniform(double a, double b);
	double normal(double mu, double sigma);
	double rademacher();

	void uniform(GenericVector & v, double a, double b);
	void normal(GenericVector & v, double sigma, bool zero_out);
	void rademacher(GenericVector & v);

private:
	PRNG<std::mt19937> eng;
	std::normal_distribution<> d_normal;
	std::uniform_real_distribution<> d_uniform;
	std::bernoulli_distribution d_bernoulli;

};
}
