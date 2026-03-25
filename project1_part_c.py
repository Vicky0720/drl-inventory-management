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
from project1_part_b import DEVICE, Policy, to_tensor


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
) -> Dict[str, Dict]:
    dqn_logs = dqn_agent.train(
        env,
        replay_buffer,
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        eval_every=eval_every,
        eval_runs=eval_runs,
    )
    ac_logs = ac_agent.train(
        env,
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        eval_every=eval_every,
        eval_runs=eval_runs,
    )
    pg_logs = pg_agent.train(
        env,
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        eval_every=eval_every,
        eval_runs=eval_runs,
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
    dqn_agent,
    ac_agent,
    pg_agent: PG_agent,
    replay_buffer,
    gamma: float = 0.99,
    demand_lambda: float = 5.0,
    max_order: int = 30,
    heuristic_search_space: Optional[Dict[str, Sequence[int]]] = None,
    episodes: int = 200,
    steps_per_episode: int = 100,
    eval_every: int = 20,
    eval_runs: int = 20,
    seed: int = 42,
) -> Dict[str, Dict]:
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
    )

    return {
        "config": {"l": l, "p": p, "gamma": gamma, "demand_lambda": demand_lambda},
        "heuristics": heuristics,
        "drl": drl_results,
    }


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

