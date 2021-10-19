#ifndef EQOPT_ZDD_HPP
#define EQOPT_ZDD_HPP

#include "common.hpp"
#include "graph.hpp"

#include <vector>

class ZDDNode{
public:
  int lv;
  addr_t lo;
  addr_t hi;
  
  ZDDNode(int _lv, addr_t _lo, addr_t _hi): lv(_lv), lo(_lo), hi(_hi) {};
  ZDDNode(): lv(0), lo(0), hi(0) {};
          
  bool operator==(const ZDDNode& right) const {
    return lv == right.lv && lo == right.lo && hi == right.hi;
  }
  bool operator<(const ZDDNode& right) const {
    return lv > right.lv;
  }
};

class ZDD{
public:
  std::vector<ZDDNode> nodes;
  std::vector<int> vars;
  std::vector<int> invvars;
  size_t dim;
  
  ZDD() {};
  
  inline size_t size() const{
    return nodes.size();
  }
  inline size_t numvars() const{
    return vars.size();
  }
  inline int var(int i) const{
    return vars[i];
  }
  inline addr_t loid(addr_t i) const{
    return nodes[i].lo;
  }
  inline addr_t hiid(addr_t i) const{
    return nodes[i].hi;
  }
  inline int lv(addr_t i) const{
    return nodes[i].lv;
  }
  
  bool readfromFile(const char *filename);
  int readOrderfromFile(const char *filename, const Graph& g);
  Real linearOptim(const RealV& w, RealV& x) const;              // non-active (since non-differentiable)
  void softMinimum(const aRealV& w, aRealV& x, Real eta) const;  // active version
  void softMinimum(const RealV& w, RealV& x, Real eta) const;    // non-active version
};

#endif // EQOPT_ZDD_HPP
