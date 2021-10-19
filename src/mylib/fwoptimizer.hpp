#ifndef EQOPT_FWOPTIMIZER_HPP
#define EQOPT_FWOPTIMIZER_HPP

#include "common.hpp"
#include "zdd.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <unordered_map>

template <typename GRAD>
class FWOptimizer{
public:
  GRAD* Gradfunc;
  const ZDD* Z;
  Real eta = 10.0;
  Real tol = 1e-7;
  Real exponent = 1.0;
  Real etamin = 0.0;
  int maxK = 200;
  bool printgap = false;
  bool etaconst = false;
  bool timecalc = false;
  bool fwgapcomp = true;
  size_t dim;
  std::chrono::system_clock::time_point tbase;
  
  FWOptimizer(GRAD *_Gradfunc, ZDD *_Z): Gradfunc(_Gradfunc), Z(_Z) {dim = Z->numvars();};
  FWOptimizer() {};
  
  ~FWOptimizer() {
    Gradfunc->destructParams();
  }
  
  void seteta(Real _eta){
    eta = _eta;
  }
  void settol(Real _tol){
    tol = _tol;
  }
  void setmaxK(int _maxK){
    maxK = _maxK;
  }
  void setexponent(Real _exponent){
    exponent = _exponent;
  }
  void setetamin(Real _etamin){
    etamin = _etamin;
  }
  void setprintgap(bool _printgap){
    printgap = _printgap;
  }
  void setconsteta(bool _etaconst){
    etaconst = _etaconst;
  }
  void settimecalc(bool _timecalc){
    timecalc = _timecalc;
  }
  void setfwgapcomp(bool _fwgapcomp){
    fwgapcomp = _fwgapcomp;
  }
  
  // duration time calculation
  void setbasetime(){
    tbase = std::chrono::system_clock::now();
  }
  double calcdurationtime() const{
    if(timecalc){
      auto tnow = std::chrono::system_clock::now();
      return std::chrono::duration_cast<std::chrono::microseconds>(tnow - tbase).count() / 1000.0;
    }else return 0.0;
  }
  
  // solve with auto-differentiation
  Real solve(const RealV& paramsval, const RealV& x0, RealV& grad, RealV& newx, int mode) const;
  // solve without auto-differentiation
  Real NDsolve(const RealV& paramsval, const RealV& x0, RealV& newx, int mode) const;
  
  // Solve with differentiable normal Frank--Wolfe method
  template <typename xReal>
  xReal FWOptim(std::vector<xReal>& x) const;
  
  // Solve with differentiable accelerated Frank--Wolfe method
  template <typename xReal>
  xReal ACFWOptim(std::vector<xReal>& x) const;
  
  // Solve with differentiable accelerated Frank--Wolfe method ver.3
  template <typename xReal>
  xReal AC3FWOptim(std::vector<xReal>& x) const;
  
  // Solve with normal Frank--Wolfe method (non-differentiable)
  Real NDFWOptim(RealV& x) const;
  
  // Test function (calculate Jacobian of softmin function)
  //void Jacobian_ZDD_softmin(const RealV& wval, RealV& xval, RealV& jac) const;
  
private:
  mutable adept::Stack stack;
  // Calculate Frank--Wolfe gap
  Real calcFWgap(const aRealV& x) const;  // active version
  Real calcFWgap(const RealV& x) const;   // non-active version
};

template <typename GRAD>
Real FWOptimizer<GRAD>::solve(const RealV& paramsval, const RealV& x0, RealV& grad, RealV& newx, int mode) const{
  Gradfunc->setParams(paramsval);
  aRealV x(dim);
  adept::set_values(&x[0], dim, &x0[0]);
  stack.new_recording();
  
  aReal J;
  switch(mode){
    case 1: J = FWOptim(x); break;
    case 2: J = ACFWOptim(x); break;
    case 3: J = AC3FWOptim(x); break;
    default: J = 0.0; break;
  }
  
  J.set_gradient(1.0);
  stack.compute_adjoint();
  //stack.print_statements();
  Gradfunc->getGradParams(grad);
  
  newx.resize(dim);
  for(size_t i=0; i<dim; ++i) newx[i] = adept::value(x[i]);
  return adept::value(J);
}

template <typename GRAD>
Real FWOptimizer<GRAD>::NDsolve(const RealV& paramsval, const RealV& x0, RealV& newx, int mode) const{
  RealV x(x0);
  Gradfunc->setParams(paramsval, false);
  
  Real J;
  switch(mode){
    case 1: J = FWOptim(x); break;
    case 2: J = ACFWOptim(x); break;
    case 3: J = AC3FWOptim(x); break;
    case 4: J = NDFWOptim(x); break;
    default: J = 0.0; break;
  }
  
  newx.clear();
  newx.reserve(dim);
  newx.insert(newx.end(), x.begin(), x.end());
  return J;
}

template <typename GRAD>
template <typename xReal>
xReal FWOptimizer<GRAD>::FWOptim(std::vector<xReal>& x) const{
  for(int k=1; k<=maxK; ++k){
    std::vector<xReal> w(dim);
    std::vector<xReal> s(dim);
    Real gamma = 2.0 / (1.0 + k);
    // compute w = grad(F) with cost x
    Gradfunc->gfunc(x, w);
    // compute softmin with weight w
    if(etaconst) Z->softMinimum(w, s, eta);
    else         Z->softMinimum(w, s, eta*pow(k, exponent));
    // compute x_k = (1-gamma_k)*x_k-1 + gamma_k*s_k
    for(size_t i=0; i<dim; ++i){
      x[i] = (1.0 - gamma) * x[i] + gamma * s[i];
    }
    // check stopping criterion
    if(fwgapcomp){
      Real fwgap = calcFWgap(x);
      if(printgap) printf("%d,%.10lf,%.3lf\n", k, fwgap, calcdurationtime());
      if(fwgap <= tol) break;
    }else if(timecalc){
      printf("%d,%.3lf\n", k, calcdurationtime());
    }
  }
  return Gradfunc->fval(x);
}

// Slightly different version of the implementation of accelerated differentiable Frank--Wolfe;
// Its behavior is unstable; DO NOT USE!
template <typename GRAD>
template <typename xReal>
xReal FWOptimizer<GRAD>::ACFWOptim(std::vector<xReal>& x) const{
  std::vector<std::vector<xReal>> y;
  std::vector<xReal> s(dim), c(dim);
  y.resize(maxK+1);
  
  // initialize y_0 = x and y_-1 = s_0 = x_0 = 0
  y[0].resize(dim);
  for(size_t i=0; i<dim; ++i) y[0][i] = x[i];
  for(size_t i=0; i<dim; ++i) s[i] = x[i];
  for(size_t i=0; i<dim; ++i) c[i] = 0.0;
  for(int k=1; k<=maxK; ++k){
    Real gamma = 2.0 / (1.0 + k);
    // compute s_k = s_k-1 - (k-1)*y_k-2 + (2k-1)*y_k-1
    if(k != 1){
      for(size_t i=0; i<dim; ++i) s[i] += (2*k-1)*y[k-1][i] - (k-1)*y[k-2][i];
    }
    // compute w = grad(F) with cost 2/k(k+1)*s_t and c_k = c_k-1 + k*w;
    std::vector<xReal> sr(dim);
    std::vector<xReal> w(dim);
    for(size_t i=0; i<dim; ++i) sr[i] = s[i]/(0.5*k*(k+1));
    Gradfunc->gfunc(sr, w);
    for(size_t i=0; i<dim; ++i) c[i] += k*w[i];
    // compute softmin with weight c_k
    if(etaconst) Z->softMinimum(c, y[k], eta);
    else{
      Real eta_k = std::max(eta/pow(k, exponent), etamin);
      Z->softMinimum(c, y[k], eta_k);
    }
    // compute x_k = (1-gamma_k)*x_k-1 + gamma_k*y_k
    for(size_t i=0; i<dim; ++i) x[i] = (1.0-gamma)*x[i] + gamma*y[k][i];
    // check stopping criterion
    if(fwgapcomp){
      Real fwgap = calcFWgap(x);
      if(printgap) printf("%d,%.10lf,%.3lf\n", k, fwgap, calcdurationtime());
      if(fwgap <= tol) break;
    }else if(timecalc){
      printf("%d,%.3lf\n", k, calcdurationtime());
    }
  }
  return Gradfunc->fval(x);
}

template <typename GRAD>
template <typename xReal>
xReal FWOptimizer<GRAD>::AC3FWOptim(std::vector<xReal>& x) const{
  std::vector<std::vector<xReal>> y;
  std::vector<xReal> s(dim), c(dim);
  y.resize(maxK+1);
  
  // initialize y_0 = x and y_-1 = s_0 = x_0 = 0
  y[0].resize(dim);
  for(size_t i=0; i<dim; ++i) y[0][i] = x[i];
  for(size_t i=0; i<dim; ++i) s[i] = x[i];
  for(size_t i=0; i<dim; ++i) c[i] = 0.0;
  for(int k=1; k<=maxK; ++k){
    Real gamma = 2.0 / (1.0 + k);
    // compute s_k = s_k-1 - (k-1)*y_k-2 + (2k-1)*y_k-1
    if(k != 1){
      for(size_t i=0; i<dim; ++i) s[i] += (2*k-1)*y[k-1][i] - (k-1)*y[k-2][i];
    }
    // compute w = grad(F) with cost 2/k(k+1)*s_t and c_k = c_k-1 + k*w;
    std::vector<xReal> sr(dim);
    std::vector<xReal> w(dim);
    for(size_t i=0; i<dim; ++i) sr[i] = s[i]/(0.5*k*(k+1));
    Gradfunc->gfunc(sr, w);
    if(etaconst){
      for(size_t i=0; i<dim; ++i) c[i] += eta*k*w[i];
    }else{
      Real eta_k = std::max(eta/pow(k, exponent), etamin);
      for(size_t i=0; i<dim; ++i) c[i] += eta_k*k*w[i];
    }
    // compute softmin with weight c_k
    Z->softMinimum(c, y[k], 1.0);
    // compute x_k = (1-gamma_k)*x_k-1 + gamma_k*y_k
    for(size_t i=0; i<dim; ++i) x[i] = (1.0-gamma)*x[i] + gamma*y[k][i];
    // check stopping criterion
    if(fwgapcomp){
      Real fwgap = calcFWgap(x);
      if(printgap) printf("%d,%.10lf,%.3lf\n", k, fwgap, calcdurationtime());
      if(fwgap <= tol) break;
    }else if(timecalc){
      printf("%d,%.3lf\n", k, calcdurationtime());
    }
  }
  return Gradfunc->fval(x);
}

template <typename GRAD>
Real FWOptimizer<GRAD>::NDFWOptim(RealV& x) const{
  for(int k=1; k<=maxK; ++k){
    RealV w(dim);
    RealV s(dim);
    Real gamma = 2.0 / (1.0 + k);
    // compute w = grad(F) with cost x
    Gradfunc->gfunc(x, w);
    // compute min with weight w
    Z->linearOptim(w, s);
    // compute x_k = (1-gamma_k)*x_k-1 + gamma_k*s_k
    for(size_t i=0; i<dim; ++i){
      x[i] = (1.0 - gamma) * x[i] + gamma * s[i];
    }
    // check stopping criterion
    if(fwgapcomp){
      Real fwgap = calcFWgap(x);
      if(printgap) printf("%d,%.10lf,%.3lf\n", k, fwgap, calcdurationtime());
      if(fwgap <= tol) break;
    }else if(timecalc){
      printf("%d,%.3lf\n", k, calcdurationtime());
    }
  }
  return Gradfunc->fval(x);
}
/*
template <typename GRAD>
void FWOptimizer<GRAD>::Jacobian_ZDD_softmin(const RealV& wval, RealV& xval, RealV& jac) const{
  aRealV w(dim);
  adept::set_values(&w[0], dim, &wval[0]);
  stack.new_recording();
  aRealV x(dim);
  Z->softMinimum(w, x, eta);
  stack.independent(&w[0], dim);
  stack.dependent(&x[0], dim);
  jac.resize(dim*dim);
  stack.jacobian_reverse(&jac[0]);
  xval.resize(dim);
  for(size_t i=0; i<dim; ++i) xval[i] = adept::value(x[i]);
}
*/

template <typename GRAD>
Real FWOptimizer<GRAD>::calcFWgap(const aRealV& x) const{
  RealV x_na(dim);
  for(size_t i=0; i<dim; ++i) x_na[i] = adept::value(x[i]);
  RealV w_na(dim);
  Gradfunc->gfunc(x_na, w_na);
  RealV x_min(dim);
  Real iprodmin = Z->linearOptim(w_na, x_min);
  Real iprod = 0.0;
  for(size_t i=0; i<dim; ++i) iprod += w_na[i] * x_na[i];
  return iprod - iprodmin;
}

template <typename GRAD>
Real FWOptimizer<GRAD>::calcFWgap(const RealV& x) const{
  RealV w_na(dim);
  Gradfunc->gfunc(x, w_na);
  RealV x_min(dim);
  Real iprodmin = Z->linearOptim(w_na, x_min);
  Real iprod = 0.0;
  for(size_t i=0; i<dim; ++i) iprod += w_na[i] * x[i];
  return iprod - iprodmin;
}

#endif // EQOPT_FWOPTIMIZER_HPP
