#ifndef COST_INVLINEARWEIGHT_HPP
#define COST_INVLINEARWEIGHT_HPP

#include "mylib/common.hpp"
#include "mylib/graph.hpp"
#include <vector>
#include <cstdlib>
#include <cstdio>

class InvLCostW{
public:
  aRealV params;
  RealV params_na;
  size_t pdim;
  RealV weights;
  
  InvLCostW() {};
          
  // must be called before stack.new_recording()
  void setParams(const RealV& paramsval, bool gradcalc = true);
  // must be called after stack.compute_adjoint()
  void getGradParams(RealV& grad) const;
  // must be called before current stack object is destructed
  void destructParams(){
    params.clear();
    params_na.clear();
  }

  void gfunc(const aRealV& x, aRealV& res) const; // grad used in FW iteration; active version
  void gfunc(const RealV& x, RealV& res) const;   // grad used in FW iteration; non-active version
  
  aReal fval(const aRealV& x) const;  // social cost; active version
  Real fval(const RealV& x) const;    // social cost; non-active version
  
  int readWeightfromFile(const char *filename, const Graph& g);
  void setDefaultWeight(size_t num_edge);
  
  Real getWeight(size_t ind){
    return weights[ind];
  }
};

void InvLCostW::setParams(const RealV& paramsval, bool gradcalc){
  pdim = paramsval.size();
  if(gradcalc){
    params.resize(pdim);
    adept::set_values(&params[0], pdim, &paramsval[0]);
  }
  params_na.resize(pdim);
  std::copy(paramsval.begin(), paramsval.end(), params_na.begin());
}

void InvLCostW::getGradParams(RealV& grad) const{
  grad.resize(pdim);
  adept::get_gradients(&params[0], pdim, &grad[0]);
}

void InvLCostW::gfunc(const aRealV& x, aRealV& res) const{
  size_t dim = x.size();
  assert(dim == pdim);
  res.resize(dim);
  for(size_t i=0; i<dim; ++i) res[i] = weights[i] * (10.0 * x[i] / params[i] + 1.0);
}
void InvLCostW::gfunc(const RealV& x, RealV& res) const{
  size_t dim = x.size();
  assert(dim == pdim);
  res.resize(dim);
  for(size_t i=0; i<dim; ++i) res[i] = weights[i] * (10.0 * x[i] / params_na[i] + 1.0);
}

aReal InvLCostW::fval(const aRealV& x) const{
  size_t dim = x.size();
  assert(dim == pdim);
  aRealV wf(dim);
  gfunc(x, wf);
  aReal res = 0.0;
  for(size_t i=0; i<dim; ++i) res += wf[i] * x[i];
  return res;
}
Real InvLCostW::fval(const RealV& x) const{
  size_t dim = x.size();
  assert(dim == pdim);
  RealV wf(dim);
  gfunc(x, wf);
  Real res = 0.0;
  for(size_t i=0; i<dim; ++i) res += wf[i] * x[i];
  return res;
}

int InvLCostW::readWeightfromFile(const char *filename, const Graph& g){
  FILE *fp;
  if((fp = fopen(filename, "r")) == NULL){
    return 1;
  }
  weights.resize(g.numE());
  std::fill(weights.begin(), weights.end(), 1.0);
  
  int u, v;
  while(fscanf(fp, "%d%d", &u, &v) != EOF){
    int ord = g.etovar(u, v);
    if(ord < 0){
      fclose(fp);
      return 2;
    }
    fscanf(fp, "%lf", &weights[ord]);
  }
  fclose(fp);
  return 0;
}

void InvLCostW::setDefaultWeight(size_t num_edge){
  weights.resize(num_edge);
  std::fill(weights.begin(), weights.end(), 1.0);
}

#endif // COST_INVLINEARWEIGHT_HPP
