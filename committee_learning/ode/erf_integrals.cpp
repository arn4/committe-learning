#include <cmath>
#include <boost/math/constants/constants.hpp>
#include <boost/math/quadrature/trapezoidal.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/next.hpp>

template<class T>
inline T square(T x){
  return x*x;
}

using namespace boost::math;
using namespace boost::math::constants;

namespace committee_learning{
namespace erfode {
  template<typename Real>
  inline Real I2(Real C11, Real C12, Real C22) {
    return std::asin(C12/std::sqrt((1.+C11)*(1.+C22)))/pi<Real>();
  }
  
  // template<typename Real>
  // inline Real I2_noise(Real C11, Real C12, Real C22) {
  //   return 2./(pi<Real>()*std::sqrt(1+C11+C22+(C11*C22)-C12*C12));
  // }
  template<typename Real>
  inline Real I2_noise(Real C11, Real C12, Real C22) {
    return 9 - 9*C11 + 18*C12*C12 - 9*C22 + 9*C11*C22;
  }

  template<typename Real>
  inline Real I2_C12expectation(Real C11, Real C12_offset, Real C12_Znorm, Real C22, unsigned int Z_dimension) {
    Real alpha = half<Real>() * (Real(Z_dimension)-1.);

    auto f = [&](double x) {
      return I2(C11, C12_offset+C12_Znorm*x, C22) * 
             half<Real>() * ibeta_derivative(alpha, alpha, half<Real>()*(x+Real(1.)));
    };
    return quadrature::trapezoidal(f, float_next(Real(-1.)), float_prior(Real(1.)));
  }


  // template<typename Real>
  // inline Real I3(Real C11, Real C12, Real C13, Real C22, Real C23, Real C33) {
  //   Real L3 = (1+C11)*(1+C33)-C13*C13;
  //   return Real(2.)*(C23*(Real(1.)+C11)-C12*C13)/(pi<Real>()*(1+C11)*std::sqrt(L3));
  // }
  template<typename Real>
  inline Real I3(Real C11, Real C12, Real C13, Real C22, Real C23, Real C33) {
    return -18*C12*C13 + 9*C23 - 9*C11*C23 + 18*C13*C13*C23 + 18*C12*C13*C33 - 9*C23*C33 + 9*C11*C23*C33;
  }

  // template<typename Real>
  // inline Real I4(Real C11, Real C12, Real C13, Real C14, Real C22, Real C23, Real C24, Real C33, Real C34, Real C44) {
  //   Real L4 = (1+C11)*(1+C22) - C12*C12;
  //   Real L0 = L4*C34 - C23*C24*(1+C11) - C13*C14*(1+C22) + C12*C13*C24 + C12*C14*C23;
  //   Real L1 = L4*(1+C33) - C23*C23*(1+C11) - C13*C13*(1+C22) + 2*C12*C13*C23;
  //   Real L2 = L4*(1+C44) - C24*C24*(1+C11) - C14*C14*(1+C22) + 2*C12*C14*C24;
  //   return (4./(pi_sqr<Real>()*sqrt(L4)))*std::asin(L0/sqrt(L1*L2));
  // }
  template<typename Real>
  inline Real I4(Real C11, Real C12, Real C13, Real C14, Real C22, Real C23, Real C24, Real C33, Real C34, Real C44) {
    return -162*C13*C14 + 162*C13*C14*C22 + 324*C12*C14*C23 - 324*C13*C14*C23*C23 + 324*C12*C13*C24 - 162*C23*C24 + 162*C11*C23*C24 - 324*C13*C13*C23*C24 - 
            324*C14*C14*C23*C24 - 324*C13*C14*C24*C24 + 162*C13*C14*C33 - 162*C13*C14*C22*C33 - 324*C12*C14*C23*C33 - 324*C12*C13*C24*C33 + 162*C23*C24*C33 - 
            162*C11*C23*C24*C33 + 324*C14*C14*C23*C24*C33 + 324*C13*C14*C24*C24*C33 + 81*C34 - 81*C11*C34 + 162*C12*C12*C34 + 162*C13*C13*C34 + 162*C14*C14*C34 - 
            81*C22*C34 + 81*C11*C22*C34 - 162*C13*C13*C22*C34 - 162*C14*C14*C22*C34 - 648*C12*C13*C23*C34 + 162*C23*C23*C34 - 162*C11*C23*C23*C34 + 
            324*C14*C14*C23*C23*C34 - 648*C12*C14*C24*C34 + 1296*C13*C14*C23*C24*C34 + 162*C24*C24*C34 - 162*C11*C24*C24*C34 + 324*C13*C13*C24*C24*C34 - 81*C33*C34 + 
            81*C11*C33*C34 - 162*C12*C12*C33*C34 - 162*C14*C14*C33*C34 + 81*C22*C33*C34 - 81*C11*C22*C33*C34 + 162*C14*C14*C22*C33*C34 + 648*C12*C14*C24*C33*C34 - 
            162*C24*C24*C33*C34 + 162*C11*C24*C24*C33*C34 - 324*C13*C14*C34*C34 + 324*C13*C14*C22*C34*C34 + 648*C12*C14*C23*C34*C34 + 648*C12*C13*C24*C34*C34 - 
            324*C23*C24*C34*C34 + 324*C11*C23*C24*C34*C34 + 54*C34*C34*C34 - 54*C11*C34*C34*C34 + 108*C12*C12*C34*C34*C34 - 54*C22*C34*C34*C34 + 54*C11*C22*C34*C34*C34 + 162*C13*C14*C44 - 
            162*C13*C14*C22*C44 - 324*C12*C14*C23*C44 + 324*C13*C14*C23*C23*C44 - 324*C12*C13*C24*C44 + 162*C23*C24*C44 - 162*C11*C23*C24*C44 + 
            324*C13*C13*C23*C24*C44 - 162*C13*C14*C33*C44 + 162*C13*C14*C22*C33*C44 + 324*C12*C14*C23*C33*C44 + 324*C12*C13*C24*C33*C44 - 
            162*C23*C24*C33*C44 + 162*C11*C23*C24*C33*C44 - 81*C34*C44 + 81*C11*C34*C44 - 162*C12*C12*C34*C44 - 162*C13*C13*C34*C44 + 81*C22*C34*C44 - 
            81*C11*C22*C34*C44 + 162*C13*C13*C22*C34*C44 + 648*C12*C13*C23*C34*C44 - 162*C23*C23*C34*C44 + 162*C11*C23*C23*C34*C44 + 81*C33*C34*C44 - 
            81*C11*C33*C34*C44 + 162*C12*C12*C33*C34*C44 - 81*C22*C33*C34*C44 + 81*C11*C22*C33*C34*C44;
  }

  template<typename Real>
  inline Real I3_C13expectation(Real C11, Real C12, Real C13_offset, Real C13_Znorm, Real C22, Real C23, Real C33, unsigned int Z_dimension) {
    /*
      Integate over the distribution of Z = <u,v> where u,v are uniform spherical vectors.
      I'm using trapezoidal integration rule
    */

    Real alpha = half<Real>() * (Real(Z_dimension)-1.);

    auto f = [&](double x) {
      return I3(C11, C12, C13_offset+C13_Znorm*x, C22, C23, C33) * 
             half<Real>() * ibeta_derivative(alpha, alpha, half<Real>()*(x+Real(1.)));
    };
    return quadrature::trapezoidal(f, float_next(Real(-1.)), float_prior(Real(1.)));
  }

}
}