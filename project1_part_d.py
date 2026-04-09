"""
project1_part_d.py
==================
Heuristic-guided Reinforcement Learning (HuRL) improvements.

Inspired by: "HuRL: Heuristic-guided Reinforcement Learning"
(NeurIPS 2021, https://proceedings.neurips.cc/paper_files/paper/2021/file/
70d31b87bd021441e5e6bf23eb84a306-Paper.pdf)

Key ideas implemented:
  1. Heuristic-Guided Exploration  – replace ε-random actions with actions
     from the best heuristic policy during warm-up / early exploration.
  2. Heuristic-Shaped Reward       – add a potential-based shaping term
     Φ(s) = -V_heuristic(s) so that the shaped reward is
     r̃(s,a,s') = r(s,a) + γ·Φ(s') - Φ(s).
  3. HuRL_DQN_agent                – DQN that combines both tricks above.
  4. Batch evaluation helpers      – compare HuRL vs baseline on all 6 settings.
"""

import copy
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from project1_part_a import (
    Basestock,
    Cappedbasestock,
    Constantorder,
    Lostsale,
    Myopic1,
    average_discounted_cost,
    discounted_cost,
    evaluate_policy,
    run_heuristic_benchmarks,
)
from project1_part_b import (
    DEVICE,
    DQN_agent,
    Policy,
    QNetwork,
    Replaybuffer,
    TrainLogs,
    ValueNetwork,
    to_tensor,
)
from project1_part_c import PG_agent, set_global_seeds


# ---------------------------------------------------------------------------
# Helper: build the best heuristic policy function from benchmark results
# ---------------------------------------------------------------------------

def _build_best_heuristic_fn(
    env: Lostsale,
    benchmark_results: Dict[str, Dict],
) -> Callable[[Sequence[float]], int]:
    """Return a callable policy (state -> action) for the best heuristic."""
    best_name = min(
        benchmark_results.keys(),
        key=lambda n: benchmark_results[n]["metrics"]["avg_discounted_cost"],
    )
    best_params = benchmark_results[best_name]["params"]

    if best_name == "Basestock":
        return lambda state: Basestock(state=state, l=env.lead, **best_params)
    if best_name == "Cappedbasestock":
        return lambda state: Cappedbasestock(state=state, l=env.lead, **best_params)
    if best_name == "Constantorder":
        return lambda state: Constantorder(**best_params)
    # Myopic1
    return lambda state: Myopic1(
        state=state,
        p=env.p,
        c=env.c,
        h=env.h,
        l=env.lead,
        demand_lambda=env.demand_lambda,
        **best_params,
    )


# ---------------------------------------------------------------------------
# Heuristic Value Estimator  (Monte-Carlo rollout to estimate V^π_heuristic)
# ---------------------------------------------------------------------------

class HeuristicValueEstimator:
    """
    Estimates V^{π_h}(s) for a given heuristic policy π_h via Monte-Carlo
    rollouts, then caches the result in a tabular lookup over a discretised
    state grid for fast retrieval during training.

    For the lost-sales problem with moderate state spaces we can afford a
    small number of rollouts per state.  For states not in the cache we fall
    back to a short MC rollout on the fly.
    """

    def __init__(
        self,
        env: Lostsale,
        heuristic_fn: Callable[[Sequence[float]], int],
        mc_horizon: int = 200,
        mc_rollouts: int = 10,
    ) -> None:
        self.env = env
        self.heuristic_fn = heuristic_fn
        self.mc_horizon = mc_horizon
        self.mc_rollouts = mc_rollouts
        self._cache: Dict[Tuple, float] = {}

    def _mc_value(self, state: np.ndarray) -> float:
        costs = []
        for _ in range(self.mc_rollouts):
            rollout = self.env.rollout(
                self.heuristic_fn,
                steps=self.mc_horizon,
                initial_state=state,
            )
            costs.append(discounted_cost(rollout["costs"], self.env.discount))
        return float(np.mean(costs))

    def value(self, state: np.ndarray) -> float:
        key = tuple(np.round(state, 1))
        if key not in self._cache:
            self._cache[key] = self._mc_value(state)
        return self._cache[key]

    def potential(self, state: np.ndarray) -> float:
        """Φ(s) = -V^{π_h}(s)  (negative because we minimise cost)."""
        return -self.value(state)


# ---------------------------------------------------------------------------
# Heuristic-shaped reward
# ---------------------------------------------------------------------------

def heuristic_shaped_reward(
    reward: float,
    state: np.ndarray,
    next_state: np.ndarray,
    gamma: float,
    value_estimator: HeuristicValueEstimator,
) -> float:
    """
    Potential-based reward shaping (Ng et al., 1999):
        r̃ = r + γ·Φ(s') - Φ(s)
    Guarantees the same optimal policy as the original MDP.
    """
    phi_s  = value_estimator.potential(state)
    phi_sp = value_estimator.potential(next_state)
    return reward + gamma * phi_sp - phi_s


# ---------------------------------------------------------------------------
# HuRL DQN Agent
# ---------------------------------------------------------------------------

@dataclass
class HuRLTrainLogs:
    losses: List[float]
    eval_costs: List[float]
    best_eval_cost: Optional[float] = None
    heuristic_action_fraction: List[float] = None  # fraction of steps using heuristic


class HuRL_DQN_agent(DQN_agent):
    """
    DQN augmented with two HuRL tricks:
      1. Heuristic-guided exploration:  during the initial warm-up phase (and
         with probability ε_h thereafter) the agent picks the heuristic action
         instead of a random action.
      2. Potential-based reward shaping using V^{π_h}.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        hidden_sizes: Sequence[int] = (128, 128),
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 50,
        tau: float = 1.0,
        double_q: bool = True,
        grad_clip: Optional[float] = 5.0,
        # HuRL-specific
        heuristic_fn: Optional[Callable] = None,
        heuristic_eps: float = 0.5,   # initial prob of using heuristic action
        heuristic_eps_decay: float = 0.99,
        heuristic_eps_end: float = 0.0,
        use_reward_shaping: bool = True,
        value_estimator: Optional[HeuristicValueEstimator] = None,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=gamma,
            lr=lr,
            hidden_sizes=hidden_sizes,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            target_update_freq=target_update_freq,
            tau=tau,
            double_q=double_q,
            grad_clip=grad_clip,
        )
        self.heuristic_fn = heuristic_fn
        self.heuristic_eps = float(heuristic_eps)
        self.heuristic_eps_decay = float(heuristic_eps_decay)
        self.heuristic_eps_end = float(heuristic_eps_end)
        self.use_reward_shaping = use_reward_shaping
        self.value_estimator = value_estimator

    def select_action(self, state: Sequence[float], greedy: bool = False) -> int:
        if greedy:
            state_t = to_tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0)
            with torch.no_grad():
                return int(torch.argmax(self.q_net(state_t), dim=1).item())

        # Exploration with heuristic guidance
        rnd = np.random.rand()
        if rnd < self.epsilon:
            # Instead of pure random: use heuristic with prob heuristic_eps
            if self.heuristic_fn is not None and np.random.rand() < self.heuristic_eps:
                return int(self.heuristic_fn(state))
            return int(np.random.randint(self.action_dim))

        state_t = to_tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            return int(torch.argmax(self.q_net(state_t), dim=1).item())

    def train(
        self,
        env: Lostsale,
        replay_buffer: Replaybuffer,
        episodes: int = 200,
        steps_per_episode: int = 100,
        batch_size: int = 64,
        warmup_steps: int = 200,
        eval_every: int = 20,
        eval_runs: int = 20,
        eval_horizon: Optional[int] = None,
        restore_best: bool = True,
    ) -> HuRLTrainLogs:
        losses: List[float] = []
        eval_costs: List[float] = []
        heuristic_fractions: List[float] = []
        best_eval_cost: Optional[float] = None
        best_q_state = None
        best_target_state = None

        eval_horizon = steps_per_episode if eval_horizon is None else int(eval_horizon)
        total_steps = 0

        for episode in range(episodes):
            state = env.reset()
            heuristic_count = 0

            for _ in range(steps_per_episode):
                # Track whether the heuristic was used
                rnd = np.random.rand()
                used_heuristic = (
                    rnd < self.epsilon
                    and self.heuristic_fn is not None
                    and np.random.rand() < self.heuristic_eps
                )
                action = self.select_action(state)
                if used_heuristic:
                    heuristic_count += 1

                next_state, cost = env.step(state, action)
                reward = -cost

                # Optional reward shaping
                if self.use_reward_shaping and self.value_estimator is not None:
                    reward = heuristic_shaped_reward(
                        reward, state, next_state, self.gamma, self.value_estimator
                    )

                replay_buffer.push(state, action, reward, next_state, False)
                state = next_state
                total_steps += 1

                if len(replay_buffer) >= max(batch_size, warmup_steps):
                    batch = replay_buffer.sample(batch_size)
                    loss = self.update(batch)
                    losses.append(loss)

            heuristic_fractions.append(heuristic_count / steps_per_episode)

            # Decay heuristic epsilon
            self.heuristic_eps = max(
                self.heuristic_eps_end,
                self.heuristic_eps * self.heuristic_eps_decay,
            )

            if (episode + 1) % eval_every == 0:
                metrics = self.evaluate(env, horizon=eval_horizon, n_sim=eval_runs)
                eval_cost = metrics["avg_discounted_cost"]
                eval_costs.append(eval_cost)
                if best_eval_cost is None or eval_cost < best_eval_cost:
                    best_eval_cost = float(eval_cost)
                    if restore_best:
                        best_q_state = copy.deepcopy(self.q_net.state_dict())
                        best_target_state = copy.deepcopy(self.target_q_net.state_dict())

        if restore_best and best_q_state is not None:
            self.q_net.load_state_dict(best_q_state)
            self.target_q_net.load_state_dict(best_target_state)

        return HuRLTrainLogs(
            losses=losses,
            eval_costs=eval_costs,
            best_eval_cost=best_eval_cost,
            heuristic_action_fraction=heuristic_fractions,
        )


# ---------------------------------------------------------------------------
# HuRL PG Agent (REINFORCE with heuristic baseline + guided exploration)
# ---------------------------------------------------------------------------

class HuRL_PG_agent(PG_agent):
    """
    REINFORCE where:
      - the moving-average baseline is replaced by the heuristic value function
        V^{π_h}(s) (a stronger, state-dependent baseline).
      - during exploration the agent sometimes samples from the heuristic
        policy instead of its own stochastic policy.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        hidden_sizes: Sequence[int] = (128, 128),
        entropy_coef: float = 1e-3,
        heuristic_fn: Optional[Callable] = None,
        value_estimator: Optional[HeuristicValueEstimator] = None,
        heuristic_mix: float = 0.3,       # prob of picking heuristic action
        heuristic_mix_decay: float = 0.98,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=gamma,
            lr=lr,
            hidden_sizes=hidden_sizes,
            use_baseline=True,
            entropy_coef=entropy_coef,
        )
        self.heuristic_fn = heuristic_fn
        self.value_estimator = value_estimator
        self.heuristic_mix = float(heuristic_mix)
        self.heuristic_mix_decay = float(heuristic_mix_decay)

    def _returns_from_rewards(self, rewards: List[float], states: Optional[List[np.ndarray]] = None) -> torch.Tensor:
        returns = []
        running = 0.0
        for reward in reversed(rewards):
            running = reward + self.gamma * running
            returns.append(running)
        returns.reverse()
        returns_t = to_tensor(returns)

        if self.value_estimator is not None and states is not None:
            # Use heuristic value as baseline (state-dependent)
            baselines = to_tensor([self.value_estimator.value(s) for s in states])
            returns_t = returns_t - baselines
        elif self.use_baseline:
            episode_mean = float(returns_t.mean().item())
            self.return_baseline = (
                self.baseline_momentum * self.return_baseline
                + (1.0 - self.baseline_momentum) * episode_mean
            )
            returns_t = returns_t - self.return_baseline
        return returns_t

    def train(
        self,
        env: Lostsale,
        episodes: int = 200,
        steps_per_episode: int = 100,
        eval_every: int = 20,
        eval_runs: int = 20,
    ) -> Dict[str, List[float]]:
        policy_losses: List[float] = []
        eval_costs: List[float] = []

        for episode in range(episodes):
            state = env.reset()
            log_probs: List[torch.Tensor] = []
            entropies: List[torch.Tensor] = []
            rewards: List[float] = []
            episode_states: List[np.ndarray] = []

            for _ in range(steps_per_episode):
                episode_states.append(state.copy())

                # Heuristic-guided exploration: sometimes follow heuristic
                if self.heuristic_fn is not None and np.random.rand() < self.heuristic_mix:
                    action = int(self.heuristic_fn(state))
                    # Still compute log_prob for gradient
                    state_t = to_tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0)
                    dist = self.policy.distribution(state_t)
                    action_t = torch.tensor([action], device=DEVICE)
                    log_prob = dist.log_prob(action_t).squeeze(0)
                    entropy = dist.entropy().squeeze(0)
                else:
                    action, log_prob, entropy = self.select_action(state, greedy=False)

                next_state, cost = env.step(state, action)
                reward = -cost
                state = next_state
                log_probs.append(log_prob)
                entropies.append(entropy)
                rewards.append(float(reward))

            returns = self._returns_from_rewards(rewards, episode_states)
            log_probs_t = torch.stack(log_probs)
            entropies_t = torch.stack(entropies)

            loss = -(log_probs_t * returns.detach()).mean() - self.entropy_coef * entropies_t.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            policy_losses.append(float(loss.item()))

            # Decay heuristic mixing probability
            self.heuristic_mix = max(0.0, self.heuristic_mix * self.heuristic_mix_decay)

            if (episode + 1) % eval_every == 0:
                metrics = self.evaluate(env, horizon=steps_per_episode, n_sim=eval_runs)
                eval_costs.append(metrics["avg_discounted_cost"])

        return {"policy_losses": policy_losses, "eval_costs": eval_costs}


# ---------------------------------------------------------------------------
# Convenience: run a single setting with HuRL agents
# ---------------------------------------------------------------------------

def run_hurl_setting(
    l: int,
    p: float,
    gamma: float = 0.99,
    demand_lambda: float = 5.0,
    max_order: int = 30,
    episodes: int = 200,
    steps_per_episode: int = 100,
    eval_every: int = 20,
    eval_runs: int = 20,
    seed: int = 42,
    heuristic_search_space: Optional[Dict] = None,
    use_reward_shaping: bool = True,
    mc_horizon: int = 100,
    mc_rollouts: int = 5,
    dqn_kwargs: Optional[Dict] = None,
    pg_kwargs: Optional[Dict] = None,
    replay_capacity: int = 10000,
) -> Dict:
    """
    Run HuRL-DQN and HuRL-PG on one (l, p) setting and return results dict
    comparable with run_single_setting() output.
    """
    set_global_seeds(seed)
    env = Lostsale(
        buffer_size=10,
        gamma=gamma,
        p=p,
        l=l,
        demand_lambda=demand_lambda,
        max_order=max_order,
        seed=seed,
    )

    search_cfg = heuristic_search_space or {
        "S_values": range(0, max_order + 1),
        "r_values": range(0, max_order + 1),
    }

    # Step 1: run heuristic benchmarks
    heuristics = run_heuristic_benchmarks(
        env,
        S_values=search_cfg["S_values"],
        r_values=search_cfg["r_values"],
        horizon=1000,
        n_sim=max(eval_runs, 20),
        max_order=max_order,
    )

    best_heuristic_fn = _build_best_heuristic_fn(env, heuristics)
    state_dim = env.lead + 1
    action_dim = env.max_order + 1

    # Step 2: build value estimator (optional reward shaping)
    value_est = None
    if use_reward_shaping:
        value_est = HeuristicValueEstimator(
            env, best_heuristic_fn, mc_horizon=mc_horizon, mc_rollouts=mc_rollouts
        )

    # Step 3: HuRL-DQN
    replay_buffer = Replaybuffer(capacity=replay_capacity, state_dim=state_dim)
    hurl_dqn = HuRL_DQN_agent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=gamma,
        heuristic_fn=best_heuristic_fn,
        heuristic_eps=0.6,
        heuristic_eps_decay=0.99,
        use_reward_shaping=use_reward_shaping,
        value_estimator=value_est,
        lr=5e-4,
        hidden_sizes=(256, 256),
        epsilon_decay=0.997,
        target_update_freq=100,
        tau=0.02,
        double_q=True,
        grad_clip=5.0,
        **(dqn_kwargs or {}),
    )
    dqn_logs = hurl_dqn.train(
        env,
        replay_buffer,
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        batch_size=128,
        warmup_steps=300,
        eval_every=eval_every,
        eval_runs=eval_runs,
        restore_best=True,
    )

    # Step 4: HuRL-PG
    hurl_pg = HuRL_PG_agent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=gamma,
        heuristic_fn=best_heuristic_fn,
        value_estimator=value_est,
        heuristic_mix=0.3,
        lr=5e-4,
        hidden_sizes=(256, 256),
        entropy_coef=5e-3,
        **(pg_kwargs or {}),
    )
    pg_logs = hurl_pg.train(
        env,
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        eval_every=eval_every,
        eval_runs=eval_runs,
    )

    return {
        "config": {
            "l": l, "p": p, "gamma": gamma,
            "demand_lambda": demand_lambda, "max_order": max_order,
            "episodes": episodes, "steps_per_episode": steps_per_episode,
            "seed": seed, "use_reward_shaping": use_reward_shaping,
        },
        "heuristics": heuristics,
        "drl": {
            "HuRL_DQN": {
                "train_logs": {
                    "losses": dqn_logs.losses,
                    "eval_costs": dqn_logs.eval_costs,
                    "heuristic_action_fraction": dqn_logs.heuristic_action_fraction,
                },
                "final_metrics": hurl_dqn.evaluate(env, horizon=1000, n_sim=max(eval_runs, 20)),
            },
            "HuRL_PG": {
                "train_logs": pg_logs,
                "final_metrics": hurl_pg.evaluate(env, horizon=1000, n_sim=max(eval_runs, 20)),
            },
        },
    }


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------

def compare_hurl_vs_baseline(
    baseline_result: Dict,
    hurl_result: Dict,
) -> List[Dict]:
    """
    Produce a flat list of comparison rows given outputs from
    run_single_setting() (baseline) and run_hurl_setting() (HuRL).
    """
    rows = []
    cfg = baseline_result["config"]

    # Heuristic rows (shared)
    for name, payload in baseline_result["heuristics"].items():
        rows.append({
            "l": cfg["l"], "p": cfg["p"],
            "method": name, "family": "heuristic",
            "avg_discounted_cost": payload["metrics"]["avg_discounted_cost"],
            "std": payload["metrics"].get("avg_discounted_cost_std", float("nan")),
        })

    # Baseline DRL rows
    for name, payload in baseline_result["drl"].items():
        rows.append({
            "l": cfg["l"], "p": cfg["p"],
            "method": f"{name} (baseline)", "family": "drl_baseline",
            "avg_discounted_cost": payload["final_metrics"]["avg_discounted_cost"],
            "std": payload["final_metrics"].get("avg_discounted_cost_std", float("nan")),
        })

    # HuRL DRL rows
    for name, payload in hurl_result["drl"].items():
        rows.append({
            "l": cfg["l"], "p": cfg["p"],
            "method": name, "family": "drl_hurl",
            "avg_discounted_cost": payload["final_metrics"]["avg_discounted_cost"],
            "std": payload["final_metrics"].get("avg_discounted_cost_std", float("nan")),
        })

    return rows
