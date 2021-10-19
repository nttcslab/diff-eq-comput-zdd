#include "common.hpp"
#include "graph.hpp"
#include "zdd.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <unordered_map>

bool ZDD::readfromFile(const char *filename){
  FILE *fp;
  std::unordered_map<addr_t, addr_t> IDToPos;
  if((fp = fopen(filename, "r")) == NULL){
    return false;
  }
  nodes.emplace_back(-1, -1, -1); // pos = 0 -> FALSE
  nodes.emplace_back(-1, -1, -1); // pos = 1 -> TRUE
  addr_t pos = 2;
  char buf[1024];
  while(fgets(buf, 1023, fp), buf[0] != '.'){
    addr_t ID;
    int _lv;
    char s1[16],s2[16];
    sscanf(buf, "%lld %d %s %s", &ID, &_lv, s1, s2); --_lv;
    addr_t _lo, _hi;
    IDToPos.emplace(ID, pos);
    switch(s1[0]){
    case 'T': _lo = 1; break;
    case 'B': _lo = 0; break;
    default: _lo = IDToPos.at(atoll(s1)); break;
    }
    switch(s2[0]){
    case 'T': _hi = 1; break;
    case 'B': _hi = 0; break;
    default: _hi = IDToPos.at(atoll(s2)); break;
    }
    nodes.emplace_back(_lv, _lo, _hi);
    ++pos;
  }
  fclose(fp);
  nodes.shrink_to_fit();
  return true;
}

int ZDD::readOrderfromFile(const char *filename, const Graph& g){
  FILE *fp;
  if((fp = fopen(filename, "r")) == NULL){
    return 1;
  }
  int u, v;
  while(fscanf(fp, "%d%d", &u, &v) != EOF){
    int ord = g.etovar(u, v);
    if(ord < 0){
      fclose(fp);
      return 2;
    }
    vars.emplace_back(ord);
  }
  fclose(fp);
  vars.shrink_to_fit();
  
  size_t siz = vars.size();
  invvars.resize(siz);
  for(size_t i=0; i<siz; ++i){
    invvars[vars[i]] = i;
  }
  return 0;
}

Real ZDD::linearOptim(const RealV& w, RealV& x) const{
  size_t leng = nodes.size();
  size_t dim = vars.size();
  assert(dim == w.size());
  RealV dp(leng);
  std::vector<char> br(leng);
  dp[0] = 1.0e15;
  dp[1] = 0.0;
  for(size_t i=2; i<leng; ++i){
    Real lopt = dp[nodes[i].lo];
    Real hopt = dp[nodes[i].hi] + w[vars[nodes[i].lv]];
    if(lopt < hopt){
      dp[i] = lopt;
      br[i] = 0;
    }else{
      dp[i] = hopt;
      br[i] = 1;
    }
  }
  int now = leng-1;
  x.resize(dim);
  for(size_t i=0; i<dim; ++i) x[i] = 0.0;
  
  while(now >= 2){
    if(br[now]){
      x[vars[nodes[now].lv]] = 1.0;
      now = nodes[now].hi;
    }else now = nodes[now].lo;
  }
  return dp[leng-1];
}

void ZDD::softMinimum(const aRealV& w, aRealV& x, Real eta) const{
  size_t leng = nodes.size();
  size_t dim = vars.size();
  assert(dim == w.size());
  x.resize(dim);
  for(size_t i=0; i<dim; ++i) x[i] = 0.0;
  aRealV wdeta(dim);
  for(size_t i=0; i<dim; ++i) wdeta[i] = w[vars[i]] * eta;
  
  //backward calculation 
  aRealV lbdp(leng);
  lbdp[0] = 0.0;
  lbdp[1] = 0.0;
  for(size_t i=2; i<leng; ++i){
    if(!nodes[i].lo){
      lbdp[i] = lbdp[nodes[i].hi] - wdeta[nodes[i].lv];
    }else{
      lbdp[i] = logsumexp(lbdp[nodes[i].hi] - wdeta[nodes[i].lv], lbdp[nodes[i].lo]);
    }
  }
  
  //forward calculation
  aRealV fdp(leng);
  for(size_t i=0; i<leng-2; ++i) fdp[i] = 0.0;
  fdp[leng-1] = 1.0;
  for(size_t i=leng-1; i>=2; --i){
    aReal p0 = 0.0;
    if(nodes[i].lo) p0 = exp(lbdp[nodes[i].lo] - lbdp[i]);
    fdp[nodes[i].lo] += fdp[i] * p0;
    fdp[nodes[i].hi] += fdp[i] * (1.0 - p0);
    x[vars[nodes[i].lv]] += fdp[i] * (1.0 - p0);
  }
}

void ZDD::softMinimum(const RealV& w, RealV& x, Real eta) const{
  size_t leng = nodes.size();
  size_t dim = vars.size();
  assert(dim == w.size());
  x.resize(dim);
  for(size_t i=0; i<dim; ++i) x[i] = 0.0;
  RealV wdeta(dim);
  for(size_t i=0; i<dim; ++i) wdeta[i] = w[vars[i]] * eta;
  
  //backward calculation 
  RealV lbdp(leng);
  lbdp[0] = 0.0;
  lbdp[1] = 0.0;
  for(size_t i=2; i<leng; ++i){
    if(!nodes[i].lo){
      lbdp[i] = lbdp[nodes[i].hi] - wdeta[nodes[i].lv];
    }else{
      lbdp[i] = logsumexp(lbdp[nodes[i].hi] - wdeta[nodes[i].lv], lbdp[nodes[i].lo]);
    }
  }
  
  //forward calculation
  RealV fdp(leng);
  for(size_t i=0; i<leng-2; ++i) fdp[i] = 0.0;
  fdp[leng-1] = 1.0;
  for(size_t i=leng-1; i>=2; --i){
    Real p0 = 0.0;
    if(nodes[i].lo) p0 = exp(lbdp[nodes[i].lo] - lbdp[i]);
    fdp[nodes[i].lo] += fdp[i] * p0;
    fdp[nodes[i].hi] += fdp[i] * (1.0 - p0);
    x[vars[nodes[i].lv]] += fdp[i] * (1.0 - p0);
  }
}