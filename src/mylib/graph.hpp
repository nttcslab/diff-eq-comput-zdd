#ifndef EQOPT_GRAPH_HPP
#define EQOPT_GRAPH_HPP

#include "common.hpp"

#include <unordered_map>
#include <vector>
#include <utility>

using Edge = std::pair<int, int>;

class Graph{
public:
  int n;
  int m;
  std::vector<Edge> e;
  std::unordered_map<std::pair<int, int>, addr_t, HashPI> etopos;
  
  Graph(int _n): n(_n), m(0) {};
  Graph(): n(0), m(0) {};
  
  int numE() const{
    return m;
  }
  int numV() const{
    return n;
  }
  int etovar(int u, int v) const{
    if(u > v) std::swap(u, v);
    auto key = std::make_pair(u, v);
    auto it = etopos.find(key);
    if(it == etopos.end()) return -1;
    else return it->second;
  }
  
  bool readfromFile(const char *filename);
};

#endif // EQOPT_GRAPH_HPP
