import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


ArrayLike = np.ndarray
PolicyFn = Callable[[ArrayLike], int]


def _ensure_int_action(action: float) -> int:
    return max(0, int(round(float(action))))


def discounted_cost(costs: Sequence[float], gamma: float) -> float:
    return sum((gamma**t) * float(cost) for t, cost in enumerate(costs))


def average_discounted_cost(costs: Sequence[float], gamma: float) -> float:
    return (1.0 - gamma) * discounted_cost(costs, gamma)


def poisson_cdf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    term = math.exp(-lam)
    total = term
    for i in range(1, k + 1):
        term *= lam / i
        total += term
    return min(1.0, total)


def poisson_tail_strict_greater(k: int, lam: float) -> float:
    return 1.0 - poisson_cdf(k, lam)


@dataclass
class BufferSample:
    states: ArrayLike
    actions: ArrayLike
    costs: ArrayLike
    next_states: ArrayLike
    done: bool


class Lostsale:
    """Lost-sales inventory environment for periodic-review replenishment."""

    def __init__(
        self,
        buffer_size: int = 10,
        gamma: float = 0.99,
        p: float = 4.0,
        l: int = 2,
        demand_lambda: float = 5.0,
        c: float = 0.0,
        h: float = 1.0,
        max_order: int = 30,
        seed: Optional[int] = None,
    ) -> None:
        self.buffer_size = int(buffer_size)
        self.discount = float(gamma)
        self.p = float(p)
        self.lead = int(l)
        self.demand_lambda = float(demand_lambda)
        self.c = float(c)
        self.h = float(h)
        self.max_order = int(max_order)
        self.rng = np.random.default_rng(seed)
        self.ini_state = np.zeros(self.lead + 1, dtype=float)

    def reset(self, state: Optional[Sequence[float]] = None) -> ArrayLike:
        if state is None:
            return self.ini_state.copy()
        arr = np.asarray(state, dtype=float)
        if arr.shape != (self.lead + 1,):
            raise ValueError(f"Expected state shape {(self.lead + 1,)}, got {arr.shape}")
        return arr.copy()

    def sample_demand(self) -> int:
        return int(self.rng.poisson(self.demand_lambda))

    def inventory_position_base_stock(self, state: Sequence[float]) -> float:
        state = np.asarray(state, dtype=float)
        if self.lead <= 1:
            return float(state[0])
        return float(state[0] + np.sum(state[2:]))

    def inventory_position_capped(self, state: Sequence[float]) -> float:
        state = np.asarray(state, dtype=float)
        return float(state[0] + np.sum(state[1:]))

    def step(
        self,
        state: Sequence[float],
        action: float,
        demand: Optional[int] = None,
    ) -> Tuple[ArrayLike, float]:
        state = np.asarray(state, dtype=float)
        action = min(_ensure_int_action(action), self.max_order)
        demand = self.sample_demand() if demand is None else int(demand)

        if self.lead == 0:
            incoming = action
            next_pipeline = np.zeros(0, dtype=float)
        else:
            incoming = float(state[1])
            next_pipeline = np.zeros(self.lead, dtype=float)
            if self.lead > 1:
                next_pipeline[:-1] = state[2:]
            next_pipeline[-1] = action

        available = float(state[0] + incoming)
        end_inventory = max(available - demand, 0.0)
        lost_sales = max(demand - available, 0.0)
        cost = self.c * action + self.h * end_inventory + self.p * lost_sales

        next_state = np.zeros(self.lead + 1, dtype=float)
        next_state[0] = end_inventory
        if self.lead > 0:
            next_state[1:] = next_pipeline
        return next_state, float(cost)

    def rollout(
        self,
        policy_fn: PolicyFn,
        steps: int = 1000,
        initial_state: Optional[Sequence[float]] = None,
        return_demands: bool = False,
    ) -> Dict[str, ArrayLike]:
        state = self.reset(initial_state)
        states = [state.copy()]
        actions: List[int] = []
        costs: List[float] = []
        demands: List[int] = []

        for _ in range(int(steps)):
            action = min(_ensure_int_action(policy_fn(state.copy())), self.max_order)
            demand = self.sample_demand()
            next_state, cost = self.step(state, action, demand)
            actions.append(action)
            costs.append(cost)
            demands.append(demand)
            states.append(next_state.copy())
            state = next_state

        result = {
            "states": np.asarray(states[:-1], dtype=float),
            "actions": np.asarray(actions, dtype=int),
            "costs": np.asarray(costs, dtype=float),
            "next_states": np.asarray(states[1:], dtype=float),
        }
        if return_demands:
            result["demands"] = np.asarray(demands, dtype=int)
        return result

    def generate_buffers(
        self,
        policy_fn: PolicyFn,
        num_buffers: int = 100,
        initial_state: Optional[Sequence[float]] = None,
    ) -> List[BufferSample]:
        steps = int(num_buffers) * self.buffer_size
        rollout = self.rollout(policy_fn, steps=steps, initial_state=initial_state)
        buffers: List[BufferSample] = []

        for k in range(int(num_buffers)):
            start = k * self.buffer_size
            end = (k + 1) * self.buffer_size
            done = end >= steps
            buffers.append(
                BufferSample(
                    states=rollout["states"][start:end],
                    actions=rollout["actions"][start:end],
                    costs=rollout["costs"][start:end],
                    next_states=rollout["next_states"][start:end],
                    done=done,
                )
            )
        return buffers

    def buffer_targets(self, buffer: BufferSample) -> ArrayLike:
        targets = np.zeros(len(buffer.costs), dtype=float)
        for i in range(len(buffer.costs)):
            running = 0.0
            for j in range(i, len(buffer.costs)):
                running += (self.discount ** (j - i)) * float(buffer.costs[j])
            targets[i] = running
        return targets


def Basestock(state: Sequence[float], S: int, l: Optional[int] = None, **_: Dict) -> int:
    state = np.asarray(state, dtype=float)
    lead = len(state) - 1 if l is None else int(l)
    if lead <= 1:
        inventory_position = state[0]
    else:
        inventory_position = state[0] + np.sum(state[2:])
    return max(0, int(math.ceil(float(S - inventory_position))))


def Cappedbasestock(
    state: Sequence[float],
    S: int,
    r: int,
    l: Optional[int] = None,
    **_: Dict,
) -> int:
    state = np.asarray(state, dtype=float)
    _ = len(state) - 1 if l is None else int(l)
    inventory_position = state[0] + np.sum(state[1:])
    return min(int(r), max(0, int(math.ceil(float(S - inventory_position)))))


def Constantorder(r: int, **_: Dict) -> int:
    return max(0, int(r))


def Myopic1(
    state: Sequence[float],
    p: float,
    demand_lambda: float = 5.0,
    c: float = 0.0,
    h: float = 1.0,
    l: Optional[int] = None,
    max_order: int = 100,
    **_: Dict,
) -> int:
    state = np.asarray(state, dtype=float)
    lead = len(state) - 1 if l is None else int(l)
    target_prob = (c + h) / (p + h)
    available_without_new_order = int(round(float(state[0] + np.sum(state[1:]))))
    lam = demand_lambda * (lead + 1)

    for z in range(max_order + 1):
        stockout_prob = poisson_tail_strict_greater(available_without_new_order + z, lam)
        if stockout_prob <= target_prob:
            return z
    return max_order


def make_heuristic_policy(
    heuristic_fn: Callable,
    **params: Dict,
) -> PolicyFn:
    return lambda state: heuristic_fn(state=state, **params)


def evaluate_policy(
    env: Lostsale,
    policy_fn: PolicyFn,
    horizon: int = 1000,
    n_sim: int = 100,
    initial_state: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    discounted_costs: List[float] = []
    avg_discounted_costs: List[float] = []
    mean_step_costs: List[float] = []

    for _ in range(int(n_sim)):
        rollout = env.rollout(policy_fn, steps=horizon, initial_state=initial_state)
        costs = rollout["costs"]
        discounted = discounted_cost(costs, env.discount)
        discounted_costs.append(discounted)
        avg_discounted_costs.append(average_discounted_cost(costs, env.discount))
        mean_step_costs.append(float(np.mean(costs)))

    return {
        "discounted_cost_mean": float(np.mean(discounted_costs)),
        "discounted_cost_std": float(np.std(discounted_costs)),
        "avg_discounted_cost": float(np.mean(avg_discounted_costs)),
        "avg_discounted_cost_std": float(np.std(avg_discounted_costs)),
        "mean_step_cost": float(np.mean(mean_step_costs)),
        "mean_step_cost_std": float(np.std(mean_step_costs)),
    }


def search_basestock(
    env: Lostsale,
    S_values: Sequence[int],
    horizon: int = 1000,
    n_sim: int = 100,
) -> Dict[str, object]:
    best = None
    for S in S_values:
        policy = make_heuristic_policy(Basestock, S=int(S), l=env.lead)
        metrics = evaluate_policy(env, policy, horizon=horizon, n_sim=n_sim)
        candidate = {"policy_name": "Basestock", "params": {"S": int(S)}, "metrics": metrics}
        if best is None or metrics["avg_discounted_cost"] < best["metrics"]["avg_discounted_cost"]:
            best = candidate
    return best


def search_capped_basestock(
    env: Lostsale,
    S_values: Sequence[int],
    r_values: Sequence[int],
    horizon: int = 1000,
    n_sim: int = 100,
) -> Dict[str, object]:
    best = None
    for S in S_values:
        for r in r_values:
            policy = make_heuristic_policy(Cappedbasestock, S=int(S), r=int(r), l=env.lead)
            metrics = evaluate_policy(env, policy, horizon=horizon, n_sim=n_sim)
            candidate = {
                "policy_name": "Cappedbasestock",
                "params": {"S": int(S), "r": int(r)},
                "metrics": metrics,
            }
            if best is None or metrics["avg_discounted_cost"] < best["metrics"]["avg_discounted_cost"]:
                best = candidate
    return best


def search_constant_order(
    env: Lostsale,
    r_values: Sequence[int],
    horizon: int = 1000,
    n_sim: int = 100,
) -> Dict[str, object]:
    best = None
    for r in r_values:
        policy = make_heuristic_policy(Constantorder, r=int(r))
        metrics = evaluate_policy(env, policy, horizon=horizon, n_sim=n_sim)
        candidate = {"policy_name": "Constantorder", "params": {"r": int(r)}, "metrics": metrics}
        if best is None or metrics["avg_discounted_cost"] < best["metrics"]["avg_discounted_cost"]:
            best = candidate
    return best


def evaluate_myopic1(
    env: Lostsale,
    horizon: int = 1000,
    n_sim: int = 100,
    max_order: int = 100,
) -> Dict[str, object]:
    policy = make_heuristic_policy(
        Myopic1,
        p=env.p,
        c=env.c,
        h=env.h,
        l=env.lead,
        demand_lambda=env.demand_lambda,
        max_order=max_order,
    )
    return {
        "policy_name": "Myopic1",
        "params": {"max_order": int(max_order)},
        "metrics": evaluate_policy(env, policy, horizon=horizon, n_sim=n_sim),
    }


def run_heuristic_benchmarks(
    env: Lostsale,
    S_values: Optional[Sequence[int]] = None,
    r_values: Optional[Sequence[int]] = None,
    horizon: int = 1000,
    n_sim: int = 100,
    max_order: int = 100,
) -> Dict[str, Dict[str, object]]:
    S_values = list(range(0, env.max_order + env.lead * 10 + 1)) if S_values is None else list(S_values)
    r_values = list(range(0, env.max_order + 1)) if r_values is None else list(r_values)

    results = {
        "Basestock": search_basestock(env, S_values=S_values, horizon=horizon, n_sim=n_sim),
        "Cappedbasestock": search_capped_basestock(
            env,
            S_values=S_values,
            r_values=r_values,
            horizon=horizon,
            n_sim=n_sim,
        ),
        "Constantorder": search_constant_order(env, r_values=r_values, horizon=horizon, n_sim=n_sim),
        "Myopic1": evaluate_myopic1(env, horizon=horizon, n_sim=n_sim, max_order=max_order),
    }
    return results
