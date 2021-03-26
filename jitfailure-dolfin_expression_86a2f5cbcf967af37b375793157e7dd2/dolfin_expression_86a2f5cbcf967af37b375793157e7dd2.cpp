
// Based on https://gcc.gnu.org/wiki/Visibility
#if defined _WIN32 || defined __CYGWIN__
    #ifdef __GNUC__
        #define DLL_EXPORT __attribute__ ((dllexport))
    #else
        #define DLL_EXPORT __declspec(dllexport)
    #endif
#else
    #define DLL_EXPORT __attribute__ ((visibility ("default")))
#endif

#include <dolfin/function/Expression.h>
#include <dolfin/math/basic.h>
#include <Eigen/Dense>


// cmath functions
using std::cos;
using std::sin;
using std::tan;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::cosh;
using std::sinh;
using std::tanh;
using std::exp;
using std::frexp;
using std::ldexp;
using std::log;
using std::log10;
using std::modf;
using std::pow;
using std::sqrt;
using std::ceil;
using std::fabs;
using std::floor;
using std::fmod;
using std::max;
using std::min;

const double pi = DOLFIN_PI;


namespace dolfin
{
  class dolfin_expression_86a2f5cbcf967af37b375793157e7dd2 : public Expression
  {
     public:
       std::shared_ptr<dolfin::GenericFunction> generic_function_T_;
std::shared_ptr<dolfin::GenericFunction> generic_function_T_n;


       dolfin_expression_86a2f5cbcf967af37b375793157e7dd2()
       {
            
       }

       void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const override
       {
          double T_n[3];

            generic_function_T_n->eval(Eigen::Map<Eigen::Matrix<double, 3, 1>>(T_n), x);
          double T_[3];

            generic_function_T_->eval(Eigen::Map<Eigen::Matrix<double, 3, 1>>(T_), x);
          values[0] = 0.6*T_ + 0.4*T_n;

       }

       void set_property(std::string name, double _value) override
       {

       throw std::runtime_error("No such property");
       }

       double get_property(std::string name) const override
       {

       throw std::runtime_error("No such property");
       return 0.0;
       }

       void set_generic_function(std::string name, std::shared_ptr<dolfin::GenericFunction> _value) override
       {
          if (name == "T_") { generic_function_T_ = _value; return; }          if (name == "T_n") { generic_function_T_n = _value; return; }
       throw std::runtime_error("No such property");
       }

       std::shared_ptr<dolfin::GenericFunction> get_generic_function(std::string name) const override
       {
          if (name == "T_") return generic_function_T_;          if (name == "T_n") return generic_function_T_n;
       throw std::runtime_error("No such property");
       }

  };
}

extern "C" DLL_EXPORT dolfin::Expression * create_dolfin_expression_86a2f5cbcf967af37b375793157e7dd2()
{
  return new dolfin::dolfin_expression_86a2f5cbcf967af37b375793157e7dd2;
}

