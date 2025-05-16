# SVRPBench

SVRPBench is an open and extensible benchmark for the Stochastic Vehicle Routing Problem (SVRP). It includes 500+ instances spanning small to large scales (10–1000 customers), designed to evaluate algorithms under realistic urban logistics conditions with uncertainty and operational constraints.

## Overview

Existing SVRP benchmarks often assume simplified, static environments, ignoring core elements of real-world routing such as time-dependent travel delays, uncertain customer availability, and dynamic disruptions. Our benchmark addresses these limitations by simulating urban logistics conditions with high fidelity:

- Travel times vary based on time-of-day traffic patterns, log-normally distributed delays, and probabilistic accident occurrences
- Customer time windows are sampled differently for residential and commercial clients using empirically grounded temporal distributions
- A systematic dataset generation pipeline that produces diverse, constraint-rich instances including multi-depot, multi-vehicle, and capacity-constrained scenarios

## Dataset

The benchmark dataset is available on Hugging Face:
[SVRPBench Dataset](https://huggingface.co/datasets/Yahias21/vrp_benchmark/tree/main)

The dataset includes various problem instances:
- Problem sizes: 10, 20, 50, 100, 200, 500, 1000 customers
- Variants: CVRP (Capacitated VRP), TWCVRP (Time Window Constrained VRP)
- Configurations: Single/Multi-depot, Single/Multi-vehicle

## Supported Algorithms

The benchmark includes implementations of several algorithms:
- OR-tools (Google's Operations Research tools)
- ACO (Ant Colony Optimization)
- Tabu Search
- Nearest Neighbor with 2-opt local search
- Reinforcement Learning models

## Benchmarking Results

Results compare algorithm performance across different problem sizes:

| Model    | CVRP10 | CVRP20 | CVRP50 | CVRP100 | CVRP200 | CVRP500 | CVRP1000 |
|----------|--------|--------|--------|---------|---------|---------|----------|
| OR-tools | 1.4284 | 1.6624 | 1.3793 | 1.1513  | 1.0466  | 0.8642  | -        |
| ACO      | 1.5763 | 1.7843 | 1.5120 | 1.2998  | 1.1752  | 1.0371  | 0.9254   |
| Tabu     | 1.4981 | 1.7102 | 1.4578 | 1.2214  | 1.1032  | 0.9723  | 0.8735   |
| NN+2opt  | 1.6832 | 1.8976 | 1.6283 | 1.3844  | 1.2627  | 1.1247  | 1.0123   |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
# Example of loading a dataset
from load_dataset import load_vrp_dataset

# Load a CVRP dataset with 50 customers, single depot configuration
dataset = load_vrp_dataset('cvrp', 50, 'single_depot')

# Run evaluation
from vrp_bench.vrp_evaluation import VRPEvaluator

evaluator = VRPEvaluator()
results = evaluator.evaluate_solver(solver_class=ACOSolver, 
                                   solver_name="ACO",
                                   sizes=[50, 100])
```

## Features

- Comprehensive evaluation framework for VRP algorithms
- Realistic travel time modeling with time-dependent patterns
- Time window constraints based on empirical distributions
- Support for multi-depot and multi-vehicle scenarios
- Visualization tools for solution analysis
- Extensible architecture for adding new algorithms

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{svrbench2025,
  author = {Heakl, Ahmed and Shaaban, Yahia Salaheldin and Takáč, Martin and Lahlou, Salem and Iklassov, Zangir},
  title = {SVRPBench: A Benchmark for Stochastic Vehicle Routing Problems},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yehias21/vrp-benchmarks}}
}
``` 
