#include "common.hpp"
#include "graph.hpp"

#include <cstdio>
#include <cstdlib>
#include <utility>

bool Graph::readfromFile(const char *filename){
  FILE *fp;
  if((fp = fopen(filename, "r")) == NULL){
    return false;
  }
  m = n = 0;
  int u, v;
  while(fscanf(fp, "%d%d", &u, &v) != EOF){
    if(u > v) std::swap(u, v);
    e.emplace_back(u, v);
    etopos.emplace(std::make_pair(u, v), m);
    n = std::max(n, u);
    n = std::max(n, v);
    ++m;
  }
  fclose(fp);
  e.shrink_to_fit();
  return true;
}
