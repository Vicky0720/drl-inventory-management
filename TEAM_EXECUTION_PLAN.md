# IE5604 Project 1 Team Execution Plan

## 1. Project Goal

Build a complete DRL project for lost-sales inventory management that satisfies both the notebook and the course guidelines:

- simulate the inventory environment correctly
- implement benchmark heuristic policies
- implement three DRL families introduced in class
- compare performance systematically
- optionally improve DRL using heuristic structure
- submit a report and a runnable Jupyter Notebook

The three DRL families to cover are:

- Q-learning: use `DQN`
- Actor-Critic: use `A2C` or a standard Actor-Critic implementation
- Policy Gradient: use `REINFORCE` or REINFORCE with baseline

## 2. Recommended 3-Person Split

### Person A: Environment + Heuristics + Evaluation

Primary ownership:

- inventory environment
- demand generation
- state transition correctness
- benchmark heuristic policies
- unified evaluation metrics

Main tasks:

- implement `Lostsale`
- implement `reset`, `step`, trajectory rollout, and buffer generation
- verify state transition for `l = 2, 3, 4`
- implement:
  - `Basestock`
  - `Cappedbasestock`
  - `Constantorder`
  - `Myopic1`
- implement parameter search for heuristic policies
- implement `evaluate_policy(...)`
- produce benchmark tables for all parameter settings

Deliverables:

- a correct environment API
- heuristic search results
- benchmark performance table
- unit sanity checks for transition logic

### Person B: Value-Based and Actor-Critic DRL

Primary ownership:

- `DQN`
- `Actor-Critic`
- replay buffer
- value function networks
- training loops for these two algorithms

Main tasks:

- implement `Replaybuffer`
- implement `QNetwork`
- implement `Policy` / actor and critic networks as needed
- implement `DQN_agent`
- implement `AC_agent`
- implement target update, epsilon exploration, batch training
- save training logs:
  - episode loss
  - evaluation cost
  - convergence curves

Deliverables:

- stable `DQN`
- stable `Actor-Critic`
- plots and logs for all experiment settings

### Person C: Policy Gradient + Experiments + Report

Primary ownership:

- `Policy Gradient`
- experiment orchestration
- plotting
- report writing and final integration

Main tasks:

- implement `REINFORCE` or `REINFORCE + baseline`
- create a unified experiment runner for:
  - all `(l, p)` combinations
  - all heuristic policies
  - all DRL methods
- create result plots and summary tables
- organize report:
  - introduction
  - methodology
  - results
  - labor of division
- integrate bonus improvement section

Deliverables:

- third DRL family required by the guideline
- unified figures and comparison tables
- final report draft

## 3. Shared Interface Contract

To avoid merge conflicts, the team should agree on these interfaces first.

### Environment interface

```python
env = Lostsale(buffer_size=10, gamma=0.99, p=4, l=2, demand_lambda=5)
state = env.reset()
next_state, cost = env.step(state, action, demand=None)
trajectory = env.rollout(policy_fn, steps=1000)
buffers = env.generate_buffers(policy_fn, num_buffers=100)
```

Notes:

- use `cost`, not `reward`, in internal logic
- if RL code expects reward, define `reward = -cost`
- all agents should consume the same state format

### Policy evaluation interface

```python
metrics = evaluate_policy(env, policy_fn, horizon=1000, n_sim=100)
```

Expected output:

```python
{
    "avg_discounted_cost": ...,
    "mean_cost": ...,
    "std_cost": ...,
}
```

### Heuristic policy interface

```python
action = heuristic_fn(state, **params)
```

### Agent interface

```python
agent.select_action(state)
agent.update(batch_or_buffer)
agent.train(env, ...)
agent.evaluate(env, ...)
```

## 4. Work Sequence

The team should not code in random order. Use this sequence.

### Phase 1: Build the foundation

1. Person A finishes the environment and evaluation function.
2. Person A verifies the heuristic formulas carefully.
3. Everyone tests against the same small sanity cases.

Exit criteria:

- state transitions are correct
- heuristic outputs are reasonable
- discounted cost can be computed consistently

### Phase 2: Build baselines

1. Person A runs heuristic parameter search.
2. Person B starts `DQN`.
3. Person B starts `Actor-Critic`.
4. Person C prepares experiment runner and result templates.

Exit criteria:

- all heuristics have best parameters
- DQN can train without crashing
- AC can train without crashing

### Phase 3: Complete the third DRL family

1. Person C implements Policy Gradient.
2. Person C plugs it into the same evaluation framework.
3. Person B and C align logging format.

Exit criteria:

- all three DRL families run on the same environment
- all methods produce comparable metrics

### Phase 4: Full experiments

Run all settings:

- `(l, p) = (2,4), (2,9), (3,4), (3,9), (4,4), (4,9)`

For each setting, record:

- best heuristic result
- DQN result
- Actor-Critic result
- Policy Gradient result
- training loss curves
- evaluation cost curves

### Phase 5: Bonus improvements

Try one or two of these:

- heuristic-guided warm start
- initialize replay buffer with heuristic samples
- imitation loss from heuristic actions
- action space restriction based on inventory structure

## 5. Merge Strategy

To reduce conflict:

- Person A edits:
  - environment cells
  - heuristic cells
  - evaluation cells
- Person B edits:
  - replay buffer
  - networks
  - DQN
  - Actor-Critic
- Person C edits:
  - Policy Gradient
  - experiment runner
  - plotting
  - report assets

Do not let all three people edit the same notebook cell at the same time.

Best practice:

- keep a backup copy of the notebook before each major merge
- also keep core code mirrored in `.py` modules if possible
- merge working code into the final notebook only after local validation

## 6. Suggested Timeline

### Day 1

- agree on interfaces
- finish environment
- finish evaluation function
- finish at least 2 heuristic policies

### Day 2

- finish all heuristics
- start DQN
- start Actor-Critic

### Day 3

- finish Policy Gradient
- finish experiment runner
- start full experiments

### Day 4

- produce all tables and plots
- implement one bonus improvement
- begin report writing

### Day 5

- polish notebook
- write final report
- package code zip

## 7. What Must Be In The Final Report

According to the guideline:

- introduction
- methodology
- results
- labor of division if collaborative

For this project, the methodology section should include:

- problem formulation
- state, action, cost, and transition definition
- heuristic policy definitions
- DRL algorithms used
- network design and hyperparameters
- training and evaluation protocol

The results section should include:

- heuristic benchmark results
- DRL performance comparison
- training curves
- analysis of algorithm behavior
- bonus improvement results if attempted

## 8. Immediate Next Step

Start with Person A's module first:

1. finalize the `Lostsale` environment
2. define `cost` and `reward = -cost`
3. implement `evaluate_policy`
4. implement `Constantorder` and `Basestock`
5. verify outputs on a tiny simulation

Only after that should the DRL agents be trained.
