# Differentiable Equilibrium Computation with Decision Diagrams for Stackelberg Models of Combinatorial Congestion Games

This repository is the official implementation of the paper "Differentiable Equilibrium Computation with Decision Diagrams for Stackelberg Models of Combinatorial Congestion Games" accepted at [Thirty-fifth Conference on Neural Information Processing Systems (NeurIPS2021)](https://nips.cc/Conferences/2021). Full version of the paper is available at [arXiv](https://arxiv.org/abs/2110.01773).

## Requirements

All codes are written in C++11 language with [Adept](http://www.met.reading.ac.uk/clouds/adept/) C++ autodiff library. Before building our codes, you should install Adept with a higher level of optimization such as `-g -O3`. For the installation of Adept, please see [Adept's documentation](http://www.met.reading.ac.uk/clouds/adept/documentation.html).

If your environment has CMake version >=3.8, you can build all codes with the following command:

```shell
source BUILD
```

After running this, all binaries will be placed on the current directory.

_Note:_ All bulit binaries and intermediate files can be removed with `source REMOVE` command.

### Verified environments

We verified that the building process of our codes and the commands presented below worked fine in the following macOS and Linux environments:

- macOS Big Sur 11.2.1 + Apple clang 12.0.0
- CentOS 7.3 + gcc 4.8.5

[Adept](http://www.met.reading.ac.uk/clouds/adept/) 2.0.5 was installed with above C++ compilers for both environments. 

Windows (VC++) environment may not be able to build our codes since  `unistd.h` is used for reading command line options.

## License

This software is released under the NTT license, see `LICENSE.txt`.

## How to reproduce experimental results

After building our code, 8 binaries are generated with two keywords: `inv` and `exp`. `inv` refers to the fractional cost scenario, i.e., $c_i(y_i;\boldsymbol{\theta})=d_i(1+C\times y_i/(\theta_i+1))$, whereas `exp` refers to the exponential cost scenario, i.e., $c_i(y_i;\boldsymbol{\theta})=d_i(1+C\times y_i\exp(-\theta_i))$. 

- `./inner_eqopt_<cost>`: check the convergence of equilibrium computation with various algorithms and parameters.
- `./eqopt_<cost>`:  perform Stackelberg model computation (i.e., solve leader's problem) with our proposed method.
- `./eqopt_<cost>_baseline`: perform Stackelberg model computation with the baseline method.
- `./eqopt_<cost>_exsearch`: perform Stackelberg model computation by an exhaustive search (Note: this program is only applicable for the toy graph described later).

All data used in our experiments are in `data/` directory. This directory includes the following graph and ZDD data:
- att48 (ATT in our paper), `graphname=att48`
- dantzig42 (DANTZIG), `graphname=dantzig42`
- Uninett (UNINETT), `graphname=Uninett2011`
- TW Telecom (TW), `graphname=Tw`
- Toy graph (toy), `graphname=toy`

Graph data of att48 and dantzig42 were retrieved from [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/), and those of Uninett and TW Telecom were retrieved from [Internet Topology Zoo](http://www.topology-zoo.org/index.html). ZDD data were generated by [Graphillion](https://github.com/takemaru/graphillion).

- `<graphname>_graph.txt`: the edge list of the graph.
- `<graphname>_distnormalized.txt`: the edge list of the graph with normalized weights.
- `<graphname>_terminals.txt`: (for Uninett and TW Telecom) the list of terminals used in the experiment. 
- `<graphname>_zdd.txt`: the ZDD representing all Hamiltonian-cycles for att48 and dantzig, that representing all Steiner trees connecting the terminals specified in `<graphname>_terminals.txt` for Uninett and TW Telecom, or that representing all simple 1-to-4 paths for toy graph.
- `<graphname>_order.txt`: the variable order (i.e. traversal order of edges) of the ZDD described in `<graphname>_zdd.txt`.
- `<graphname>_value_<cost>_initial.txt`: the initial parameter values used in the experiments for observing empirical convergence.

### Empirical convergence of equilibrium computation (Sections 4.1 and C.1)

To check the convergence of equilibrium computation (i.e., check the Frank--Wolfe gap as a function of #iteration or time), run the following commands:

```shell
./inner_eqopt_<cost> -g <graph_file> -o <order_file> -z <zdd_file> -w <weight_file> -v <value_file> (options) -t 0
./inner_eqopt_<cost> -g <graph_file> -o <order_file> -z <zdd_file> -w <weight_file> -v <value_file> (options) -u -y
```

The former command computes the Frank--Wolfe gap for each iteration, while the latter command measures the computation time without Frank--Wolfe gap computation. Concatenating these results reproduces the convergence results described in Sections 5.1 and C.1.

__List of Frank--Wolfe variants used in these experiments__:

- Proposed method (w/ A in our paper): $\eta = 0.05/0.1/0.2$
    - option: `-3 -e <eta_value> -c -i 2000`
- Non-accelerated differentiable Frank--Wolfe (w/o A): $\eta_0 = 0.1/1.0/10.0$
    - option: `-n -e <eta_value> -i 2000`
- Normal (non-differentiable) Frank--Wolfe (FW (non-diff.))
    - option: `-v -i 20000`

_Running example:_ To check the convergence of the proposed method with $\eta=0.1$ on the TW Telecom instance with the  `inv` cost scenario, run:

```shell
./inner_eqopt_inv -g data/Tw_graph.txt -o data/Tw_order.txt -z data/Tw_zdd.txt -w data/Tw_distnormalized.txt -v data/Tw_value_inv_initial.txt -3 -e 0.1 -c -i 2000 -t 0
./inner_eqopt_inv -g data/Tw_graph.txt -o data/Tw_order.txt -z data/Tw_zdd.txt -w data/Tw_distnormalized.txt -v data/Tw_value_inv_initial.txt -3 -e 0.1 -c -i 2000 -u -y
```

### Stackelberg models for designing communication networks (Sections 4.2 and C.2)

To perform Stackelberg model computation, run the following commands: 

__Proposed method:__

```shell
./eqopt_<cost> -g <graph_file> -o <order_file> -z <zdd_file> -w <weight_file> (options)
```

In the experiments described in the paper, the step size used in the outer loop (the projected gradient descent) is fixed to $5.0$, and the inner differentiable Frank--Wolfe uses $\eta=0.05/0.1/0.2$ and $T=100/200/300$. We specify them as  `-3 -e <eta_value> -c -i <T> -s 5.0 -y`.

__Baseline method:__

```shell
./eqopt_<cost>_baseline -g <graph_file> -o <order_file> -z <zdd_file> -w <weight_file> (options)
```

In the baseline method, the following heuristics is performed:

- Compute equilibrium state $\boldsymbol{y}(\boldsymbol{\theta})=(y_i)$ with current parameter $\boldsymbol{\theta}$. 
- Compute $y_{\mathrm{ave}}=\sum_i y_i/|E|$.
- Update $\boldsymbol{\theta}$ by $\theta_i \leftarrow \theta_i + \delta(y_i - y_{\mathrm{ave}})$ for all $i$. 
- Project $\boldsymbol{\theta}$ onto the simplex.
- If the above update of $\boldsymbol{\theta}$ results in the increase in the social cost, a new $\boldsymbol{\theta}$ value is chosen uniformly at random from the simplex.

We consider three values for the step size: $\delta=0.1/0.5/1.0$. We can specify this by  `-3 -e 0.1 -c -i 300 -s <delta_value> -y`.

To obtain the results of baseline methods, we run the above method 20 times and compute the average and the standard deviation.

_Note:_ With  `-p` option, we can print the values of the current parameter for each outer iteration.

_Running example:_ To perform Stackelberg model computation on the att48 instance with `inv` (fractional) cost scenario by using the proposed method with $\eta=0.1$ and $T=300$, run:

```shell
./eqopt_inv -g data/att48_graph.txt -o data/att48_order.txt -z data/att48_zdd.txt -w data/att48_distnormalized.txt -3 -e 0.1 -c -i 300 -s 5.0 -y
```

### Empirical convergence to global optimum (Section 4.3)

To reproduce empirical results in Section 4.3, run the following commands: 

__Proposed method (with projected gradient descent):__

```shell
./eqopt_<cost> -g data/toy_graph.txt -o data/toy_order.txt -z data/toy_zdd.txt -w data/toy_distnormalized.txt -3 -e 0.1 -c -i 300 -s 5.0 -y -l 50
```

_Note:_ With  `-p` option, we can print the values of the current parameter for each outer iteration.

__Exhaustive search:__

```shell
./eqopt_<cost>_exsearch -g data/toy_graph.txt -o data/toy_order.txt -z data/toy_zdd.txt -w data/toy_distnormalized.txt -3 -e 0.1 -c -i 300 -y
```
