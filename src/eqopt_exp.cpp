#include "mylib/common.hpp"
#include "mylib/graph.hpp"
#include "mylib/zdd.hpp"
#include "mylib/fwoptimizer.hpp"

#include "expcostweighted.hpp"

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <chrono>
#include <random>
#include <algorithm>

void projSimplex(RealV& pos, Real c){
  assert(c > 0);
  int dim = pos.size();
  RealV y(pos);
  for(auto &x : y) x /= c;
  std::sort(y.begin(), y.end());
  int i = dim-1;
  Real t = 0.0, sum = 0.0;
  while(1){
    sum += y[i];
    t = (sum-1.0) / (dim-i);
    if(!i || t >= y[i-1]) break;
    --i;
  }
  for(auto &x : pos) x = std::max(x-t*c, 0.0);
}

Real norm2(RealV& vec){
  Real res = 0.0;
  for(auto x : vec) res += x*x;
  return sqrt(res);
}

void print_usage(){
  fprintf(stderr, "Usage:\n");
  fprintf(stderr, " -g [graph_file_name]  : Set graph file name (mandatory)\n");
  fprintf(stderr, " -o [order_file_name]  : Set order file name (mandatory)\n");
  fprintf(stderr, " -z [zdd_file_name]    : Set zdd file name (mandatory)\n");
  fprintf(stderr, " -w [weight_file_name] : Set graph weight file name\n");  
  fprintf(stderr, " -e [eta_value]        : Set initial eta value (default: 50.0)\n");
  fprintf(stderr, " -t [tol_value]        : Set tol value (default: 1e-7)\n");
  fprintf(stderr, " -i [inner_loop_limit] : Set inner loop limit (default: 200)\n");
  fprintf(stderr, " -l [outer_loop_limit] : Set outer loop limit (default: 5000)\n");
  fprintf(stderr, " -s [step_size]        : Set outer step size (default: 1e-3)\n");
  fprintf(stderr, " -n : Use normal FW instead of accelerated FW\n");
  fprintf(stderr, " -3 : Use accelerated FW version 3\n");
  fprintf(stderr, " -c : Use constant eta\n");
  fprintf(stderr, " -x [exponent_value]   : Set exponent of varying eta (default: 1.0)\n");
  fprintf(stderr, " -b [eta_min_value]    : Set minimum value of eta (default: 0.0 (no minimum))\n");
  fprintf(stderr, " -p : Print x0 and parameter for each outer step\n");
  fprintf(stderr, " -q : Print inner FW gap\n");
  fprintf(stderr, " -y : Disable FW gap computation\n");
}

int main(int argc, char **argv){
  int opt;
  char *graph_file, *order_file, *zdd_file, *weight_file;
  Real eta_val = 50.0;
  Real tol_val = 1e-7;
  Real exponent_val = 1.0;
  Real etamin_val = 0.0;
  int iflg = 0;
  int inner_loop_num = 200;
  int outer_loop_num = 5000;
  Real step_size = 1e-3;
  bool normalfw = false;
  bool acver3 = false;
  bool etaconst = false;
  bool printparams = false;
  bool printinner = false;
  bool weightexist = false;
  bool fwgapcomp = true;
  auto rstart = std::chrono::system_clock::now();
  while((opt = getopt(argc, argv, "g:o:z:w:e:t:i:s:l:nc3x:b:pqy")) != -1){
    switch (opt) {
    case 'g': graph_file = optarg; iflg |= 1; break;
    case 'o': order_file = optarg; iflg |= 2; break;
    case 'z': zdd_file = optarg; iflg |= 4; break;
    case 'w': weight_file = optarg; weightexist = true; break;
    case 'e': eta_val = atof(optarg); break;
    case 't': tol_val = atof(optarg); break;
    case 'i': inner_loop_num = atoi(optarg); break;
    case 's': step_size = atof(optarg); break;
    case 'l': outer_loop_num = atoi(optarg); break;
    case 'n': normalfw = true; break;
    case '3': acver3 = true; break;
    case 'c': etaconst = true; break;
    case 'x': exponent_val = atof(optarg); break;
    case 'b': etamin_val = atof(optarg); break;
    case 'p': printparams = true; break;
    case 'q': printinner = true; break;
    case 'y': fwgapcomp = false; break;
    }
  }
  
  if(iflg != 7){
    print_usage();
    exit(EXIT_FAILURE);
  }
  
  if(acver3 && normalfw){
    fprintf(stderr, "[ERROR] Options -3 and -n cannot be used simultaneously.");
    print_usage();
    exit(EXIT_FAILURE);
  }
  
  Graph G;
  if(!(G.readfromFile(graph_file))){
    fprintf(stderr, "[ERROR] Cannot open graph file.\n");
    print_usage();
    exit(EXIT_FAILURE);
  }
  size_t dim = G.numE();
  ZDD Z;
  if(!(Z.readfromFile(zdd_file))){
    fprintf(stderr, "[ERROR] Cannot open zdd file: %s\n", zdd_file);
    print_usage();
    exit(EXIT_FAILURE);
  }
  if(int err = Z.readOrderfromFile(order_file, G)){
    switch(err){
      case 1: fprintf(stderr, "[ERROR] Cannot open order file: %s\n", order_file);
      case 2: fprintf(stderr, "[ERROR] Order file %s contains non-exist edge.\n", order_file);
    }
    print_usage();
    exit(EXIT_FAILURE);
  }
  
  ExpCostW GradFunc;
  if(weightexist){
    if(int err = GradFunc.readWeightfromFile(weight_file, G)){
      switch(err){
      case 1: fprintf(stderr, "[ERROR] Cannot open weight file: %s\n", weight_file);
      case 2: fprintf(stderr, "[ERROR] Weight file %s contains non-exist edge.\n", weight_file);
      }
      print_usage();
      exit(EXIT_FAILURE);
    }
  }else GradFunc.setDefaultWeight(dim);
  
  auto cstart = std::chrono::system_clock::now();
  
  size_t pdim = dim;
  RealV paramsval(pdim, 1.0);
  RealV x0(dim);
  
  if(acver3){
    RealV tmpw(dim, 0.0);
    Z.softMinimum(tmpw, x0, 1.0);
  }else{
    RealV tmpw(dim, 1.0);
    Z.linearOptim(tmpw, x0);
  }
  
  FWOptimizer<ExpCostW> FWO(&GradFunc, &Z);
  
  Real del;
  
  FWO.setmaxK(inner_loop_num);
  
  FWO.seteta(eta_val);
  FWO.settol(tol_val);
  FWO.setconsteta(etaconst);
  FWO.setexponent(exponent_val);
  FWO.setetamin(etamin_val);
  FWO.setprintgap(printinner);
  FWO.setfwgapcomp(fwgapcomp);
  
  for(int t=0; t<outer_loop_num; ++t){
    RealV grad(pdim);
    RealV newx(dim);
    
    //Print parameters (if printparams = true)
    if(printparams){
      for(const auto& v : paramsval) printf("%.10lf ", v);
      puts("");
      for(const auto& v : x0) printf("%.10lf ", v);
      puts("");
    }
    
    //Perform Frank--Wolfe and get gradient of parameters
    Real val;
    if(normalfw)    val = FWO.solve(paramsval, x0, grad, newx, 1);
    else if(acver3) val = FWO.solve(paramsval, x0, grad, newx, 3);
    else            val = FWO.solve(paramsval, x0, grad, newx, 2);
    if(printparams){
      for(const auto& v : newx) printf("%.10lf ", v);
      puts("");
    }
    double dtime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()-cstart).count();
    printf("%d,%.10lf,%.lf\n", t, val, dtime);
    
    //Update parameters
    del = step_size;    
    for(size_t i=0; i<pdim; ++i){
      paramsval[i] -= del * grad[i];
    }
    projSimplex(paramsval, 1.0 * pdim);

    //Update position
    if(!acver3){
      for(size_t i=0; i<dim; ++i) x0[i] = newx[i];
    }
  }
  
  auto cend = std::chrono::system_clock::now();
  double rtime = std::chrono::duration_cast<std::chrono::milliseconds>(cstart-rstart).count();
  double ctime = std::chrono::duration_cast<std::chrono::milliseconds>(cend-cstart).count();
  
  fprintf(stderr, "read time: %.6lf ms\n", rtime);
  fprintf(stderr, "calc time: %.6lf ms\n", ctime);
  return 0;
}