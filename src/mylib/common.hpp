#ifndef EQOPT_COMMON_HPP
#define EQOPT_COMMON_HPP

#include <cmath>
#include <cstdint>
#include <cassert>
#include <utility>

#include <adept.h>
using adept::Real;
using adept::aReal;
using RealV = std::vector<Real>;
using aRealV = std::vector<aReal>;

using addr_t = long long int;
using Bint = long long int;

constexpr uint64_t FNV_OFFSET_BASIS_64 = 14695981039346656037ULL;
constexpr uint64_t FNV_PRIME_64 = 1099511628211ULL;

// hash function of pair of ints
class HashPI{
public:
  size_t operator()(const std::pair<int, int>& x) const {
    uint64_t h = FNV_OFFSET_BASIS_64;
    h = FNV_PRIME_64* h ^ x.first;
    h = FNV_PRIME_64* h ^ x.second;
    return h;
  }
};

// calculate z := log(exp(x)+log(y))
inline aReal logsumexp(aReal x, aReal y){
  if(fabs(x-y) > 50.0f){
    return fmax(x, y);
  }else{
    return log1p(exp(y-x)) + x;
  }
}
inline Real logsumexp(Real x, Real y){
  if(fabs(x-y) > 50.0f){
    return fmax(x, y);
  }else{
    return log1p(exp(y-x)) + x;
  }
}

// calculate y := max(x, 0)
inline aReal ReLU(aReal x){
  return fmax(x, 0.0);
}
inline Real ReLU(Real x){
  return fmax(x, 0.0);
}

#endif // EQOPT_COMMON_HPP