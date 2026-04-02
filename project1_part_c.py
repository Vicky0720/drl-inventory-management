import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from project1_part_a import (
    Lostsale,
    evaluate_policy,
    make_heuristic_policy,
    run_heuristic_benchmarks,
)
from project1_part_b import AC_agent, DEVICE, DQN_agent, Policy, Replaybuffer, to_tensor


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_agents_for_env(
    env: Lostsale,
    dqn_kwargs: Optional[Dict] = None,
    ac_kwargs: Optional[Dict] = None,
    pg_kwargs: Optional[Dict] = None,
    replay_capacity: int = 10000,
):
    state_dim = env.lead + 1
    action_dim = env.max_order + 1
    dqn_agent = DQN_agent(state_dim=state_dim, action_dim=action_dim, gamma=env.discount, **(dqn_kwargs or {}))
    ac_agent = AC_agent(state_dim=state_dim, action_dim=action_dim, gamma=env.discount, **(ac_kwargs or {}))
    pg_agent = PG_agent(state_dim=state_dim, action_dim=action_dim, gamma=env.discount, **(pg_kwargs or {}))
    replay_buffer = Replaybuffer(capacity=replay_capacity, state_dim=state_dim)
    return dqn_agent, ac_agent, pg_agent, replay_buffer


class PG_agent:
    """REINFORCE with optional moving-average baseline."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        hidden_sizes: Sequence[int] = (128, 128),
        use_baseline: bool = True,
        baseline_momentum: float = 0.9,
        entropy_coef: float = 1e-3,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.use_baseline = use_baseline
        self.baseline_momentum = baseline_momentum
        self.entropy_coef = entropy_coef

        self.policy = Policy(state_dim, action_dim, hidden_sizes).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.return_baseline = 0.0

    def select_action(
        self,
        state: Sequence[float],
        greedy: bool = False,
    ) -> Tuple[int, Optional[torch.Tensor], Optional[torch.Tensor]]:
        state_t = to_tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0)
        dist = self.policy.distribution(state_t)
        if greedy:
            action = torch.argmax(dist.logits, dim=1)
            return int(action.item()), None, None
        action = dist.sample()
        log_prob = dist.log_prob(action).squeeze(0)
        entropy = dist.entropy().squeeze(0)
        return int(action.item()), log_prob, entropy

    def _returns_from_rewards(self, rewards: List[float]) -> torch.Tensor:
        returns = []
        running = 0.0
        for reward in reversed(rewards):
            running = reward + self.gamma * running
            returns.append(running)
        returns.reverse()
        returns_t = to_tensor(returns)
        if self.use_baseline:
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

            for _ in range(steps_per_episode):
                action, log_prob, entropy = self.select_action(state, greedy=False)
                next_state, cost = env.step(state, action)
                reward = -cost
                state = next_state

                log_probs.append(log_prob)
                entropies.append(entropy)
                rewards.append(float(reward))

            returns = self._returns_from_rewards(rewards)
            log_probs_t = torch.stack(log_probs)
            entropies_t = torch.stack(entropies)

            loss = -(log_probs_t * returns.detach()).mean() - self.entropy_coef * entropies_t.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            policy_losses.append(float(loss.item()))

            if (episode + 1) % eval_every == 0:
                metrics = self.evaluate(env, horizon=steps_per_episode, n_sim=eval_runs)
                eval_costs.append(metrics["avg_discounted_cost"])

        return {
            "policy_losses": policy_losses,
            "eval_costs": eval_costs,
        }

    def evaluate(self, env: Lostsale, horizon: int = 1000, n_sim: int = 100) -> Dict[str, float]:
        return evaluate_policy(env, lambda state: self.select_action(state, greedy=True)[0], horizon=horizon, n_sim=n_sim)


def train_all_drl_agents(
    env: Lostsale,
    dqn_agent,
    ac_agent,
    pg_agent: PG_agent,
    replay_buffer,
    episodes: int = 200,
    steps_per_episode: int = 100,
    eval_every: int = 20,
    eval_runs: int = 20,
    dqn_train_kwargs: Optional[Dict] = None,
    ac_train_kwargs: Optional[Dict] = None,
    pg_train_kwargs: Optional[Dict] = None,
) -> Dict[str, Dict]:
    dqn_logs = dqn_agent.train(
        env,
        replay_buffer,
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        eval_every=eval_every,
        eval_runs=eval_runs,
        **(dqn_train_kwargs or {}),
    )
    ac_logs = ac_agent.train(
        env,
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        eval_every=eval_every,
        eval_runs=eval_runs,
        **(ac_train_kwargs or {}),
    )
    pg_logs = pg_agent.train(
        env,
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        eval_every=eval_every,
        eval_runs=eval_runs,
        **(pg_train_kwargs or {}),
    )

    return {
        "DQN": {
            "train_logs": {
                "losses": dqn_logs.losses,
                "eval_costs": dqn_logs.eval_costs,
            },
            "final_metrics": dqn_agent.evaluate(env, horizon=1000, n_sim=max(eval_runs, 20)),
        },
        "ActorCritic": {
            "train_logs": ac_logs,
            "final_metrics": ac_agent.evaluate(env, horizon=1000, n_sim=max(eval_runs, 20)),
        },
        "PolicyGradient": {
            "train_logs": pg_logs,
            "final_metrics": pg_agent.evaluate(env, horizon=1000, n_sim=max(eval_runs, 20)),
        },
    }


def run_single_setting(
    l: int,
    p: float,
    dqn_agent=None,
    ac_agent=None,
    pg_agent: Optional[PG_agent] = None,
    replay_buffer=None,
    gamma: float = 0.99,
    demand_lambda: float = 5.0,
    max_order: int = 30,
    heuristic_search_space: Optional[Dict[str, Sequence[int]]] = None,
    episodes: int = 200,
    steps_per_episode: int = 100,
    eval_every: int = 20,
    eval_runs: int = 20,
    seed: int = 42,
    dqn_kwargs: Optional[Dict] = None,
    ac_kwargs: Optional[Dict] = None,
    pg_kwargs: Optional[Dict] = None,
    dqn_train_kwargs: Optional[Dict] = None,
    ac_train_kwargs: Optional[Dict] = None,
    pg_train_kwargs: Optional[Dict] = None,
    replay_capacity: int = 10000,
) -> Dict[str, Dict]:
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

    if dqn_agent is None or ac_agent is None or pg_agent is None or replay_buffer is None:
        dqn_agent, ac_agent, pg_agent, replay_buffer = build_agents_for_env(
            env,
            dqn_kwargs=dqn_kwargs,
            ac_kwargs=ac_kwargs,
            pg_kwargs=pg_kwargs,
            replay_capacity=replay_capacity,
        )

    heuristics = run_heuristic_benchmarks(
        env,
        S_values=search_cfg["S_values"],
        r_values=search_cfg["r_values"],
        horizon=1000,
        n_sim=max(eval_runs, 20),
        max_order=max_order,
    )

    drl_results = train_all_drl_agents(
        env,
        dqn_agent=dqn_agent,
        ac_agent=ac_agent,
        pg_agent=pg_agent,
        replay_buffer=replay_buffer,
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        eval_every=eval_every,
        eval_runs=eval_runs,
        dqn_train_kwargs=dqn_train_kwargs,
        ac_train_kwargs=ac_train_kwargs,
        pg_train_kwargs=pg_train_kwargs,
    )

    return {
        "config": {
            "l": l,
            "p": p,
            "gamma": gamma,
            "demand_lambda": demand_lambda,
            "max_order": max_order,
            "episodes": episodes,
            "steps_per_episode": steps_per_episode,
            "eval_every": eval_every,
            "eval_runs": eval_runs,
            "seed": seed,
            "dqn_kwargs": dqn_kwargs or {},
            "ac_kwargs": ac_kwargs or {},
            "pg_kwargs": pg_kwargs or {},
            "dqn_train_kwargs": dqn_train_kwargs or {},
            "ac_train_kwargs": ac_train_kwargs or {},
            "pg_train_kwargs": pg_train_kwargs or {},
        },
        "heuristics": heuristics,
        "drl": drl_results,
    }


def run_batch_settings(
    settings: Sequence[Tuple[int, float]],
    gamma: float = 0.99,
    demand_lambda: float = 5.0,
    max_order: int = 30,
    heuristic_search_space: Optional[Dict[str, Sequence[int]]] = None,
    episodes: int = 200,
    steps_per_episode: int = 100,
    eval_every: int = 20,
    eval_runs: int = 20,
    base_seed: int = 42,
    dqn_kwargs: Optional[Dict] = None,
    ac_kwargs: Optional[Dict] = None,
    pg_kwargs: Optional[Dict] = None,
    dqn_train_kwargs: Optional[Dict] = None,
    ac_train_kwargs: Optional[Dict] = None,
    pg_train_kwargs: Optional[Dict] = None,
    replay_capacity: int = 10000,
) -> List[Dict[str, Dict]]:
    results: List[Dict[str, Dict]] = []
    for idx, (l, p) in enumerate(settings):
        results.append(
            run_single_setting(
                l=l,
                p=p,
                gamma=gamma,
                demand_lambda=demand_lambda,
                max_order=max_order,
                heuristic_search_space=heuristic_search_space,
                episodes=episodes,
                steps_per_episode=steps_per_episode,
                eval_every=eval_every,
                eval_runs=eval_runs,
                seed=base_seed + idx,
                dqn_kwargs=dqn_kwargs,
                ac_kwargs=ac_kwargs,
                pg_kwargs=pg_kwargs,
                dqn_train_kwargs=dqn_train_kwargs,
                ac_train_kwargs=ac_train_kwargs,
                pg_train_kwargs=pg_train_kwargs,
                replay_capacity=replay_capacity,
            )
        )
    return results


def summarize_results(result: Dict[str, Dict]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    config = result["config"]
    for name, payload in result["heuristics"].items():
        rows.append(
            {
                "method": name,
                "l": config["l"],
                "p": config["p"],
                "avg_discounted_cost": payload["metrics"]["avg_discounted_cost"],
                "details": payload["params"],
            }
        )
    for name, payload in result["drl"].items():
        rows.append(
            {
                "method": name,
                "l": config["l"],
                "p": config["p"],
                "avg_discounted_cost": payload["final_metrics"]["avg_discounted_cost"],
                "details": "DRL",
            }
        )
    return rows


def plot_training_curves(
    drl_results: Dict[str, Dict],
    output_dir: Optional[str] = None,
    prefix: str = "training",
) -> Dict[str, str]:
    output_paths: Dict[str, str] = {}
    save_dir = Path(output_dir) if output_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    for name, payload in drl_results.items():
        logs = payload["train_logs"]
        plt.figure(figsize=(8, 4))
        if "losses" in logs:
            plt.plot(logs["losses"], label="loss")
        if "policy_losses" in logs:
            plt.plot(logs["policy_losses"], label="policy_loss")
        if "value_losses" in logs:
            plt.plot(logs["value_losses"], label="value_loss")
        if "eval_costs" in logs:
            plt.plot(
                np.linspace(0, max(len(logs.get("losses", logs.get("policy_losses", [1]))), 1), len(logs["eval_costs"])),
                logs["eval_costs"],
                label="eval_cost",
            )
        plt.title(f"{name} training curves")
        plt.xlabel("training step")
        plt.legend()
        plt.tight_layout()

        if save_dir is not None:
            out_path = save_dir / f"{prefix}_{name}.png"
            plt.savefig(out_path, dpi=150)
            output_paths[name] = str(out_path)
        plt.close()
    return output_paths


def save_results_json(result: Dict[str, Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def heuristic_guided_action(
    state: Sequence[float],
    env: Lostsale,
    benchmark_results: Dict[str, Dict],
) -> int:
    best_name = min(
        benchmark_results.keys(),
        key=lambda name: benchmark_results[name]["metrics"]["avg_discounted_cost"],
    )
    best = benchmark_results[best_name]
    if best_name == "Basestock":
        from project1_part_a import Basestock

        return Basestock(state=state, l=env.lead, **best["params"])
    if best_name == "Cappedbasestock":
        from project1_part_a import Cappedbasestock

        return Cappedbasestock(state=state, l=env.lead, **best["params"])
    if best_name == "Constantorder":
        from project1_part_a import Constantorder

        return Constantorder(**best["params"])
    from project1_part_a import Myopic1

    return Myopic1(
        state=state,
        p=env.p,
        c=env.c,
        h=env.h,
        l=env.lead,
        demand_lambda=env.demand_lambda,
        **best["params"],
    )
