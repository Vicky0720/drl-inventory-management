# IE5604 Project 1 Report Template

## Title

Deep Reinforcement Learning for Lost-Sales Inventory Management

## 1. Introduction

- Introduce inventory replenishment as an operations management problem.
- Explain the lost-sales setting and why it is difficult under positive lead times.
- State the project goal:
  - compare heuristic policies and DRL algorithms
  - understand algorithm behavior
  - improve DRL using problem structure where possible

## 2. Problem Formulation

- State definition:
  - on-hand inventory
  - pipeline inventory
- Action definition:
  - order quantity
- Transition dynamics:
  - periodic review
  - stochastic demand
  - lost sales
- Cost function:
  - ordering cost
  - holding cost
  - lost-sale penalty
- Evaluation metric:
  - discounted cost
  - average discounted cost

## 3. Methodology

### 3.1 Environment Simulation

- Explain the `Lostsale` environment.
- Explain demand generation and lead-time queue update.
- Explain the role of buffer size and multi-step returns.

### 3.2 Heuristic Benchmarks

- Base-stock
- Capped base-stock
- Constant-order
- Myopic-1

Describe:

- policy formulas
- parameter search ranges
- benchmark purpose

### 3.3 DRL Algorithms

- DQN
- Actor-Critic
- Policy Gradient

For each method, include:

- model architecture
- action space design
- loss function
- hyperparameters
- training setup

### 3.4 Bonus Improvement

If implemented, describe:

- heuristic-guided initialization
- warm-start replay buffer
- imitation or action-prior idea
- why it should help

## 4. Experimental Setup

- parameter settings tested:
  - `(l, p) = (2,4), (2,9), (3,4), (3,9), (4,4), (4,9)`
- training horizon
- number of simulations
- evaluation protocol
- random seeds

## 5. Results

### 5.1 Heuristic Benchmark Results

- table of best heuristic parameters
- table of best heuristic costs

### 5.2 DRL Performance

- final performance comparison table
- training curves
- convergence behavior discussion

### 5.3 Comparison and Analysis

- which methods performed best
- when heuristics were competitive
- when DRL was better
- sensitivity to lead time and penalty

### 5.4 Improvement Results

- compare before and after improvement
- explain whether convergence improved

## 6. Conclusion

- summarize the main findings
- mention key limitations
- suggest future improvements

## 7. Labor of Division

- Person A:
  - environment
  - heuristics
  - evaluation
- Person B:
  - DQN
  - Actor-Critic
  - replay buffer and neural networks
- Person C:
  - Policy Gradient
  - experiments
  - report integration

