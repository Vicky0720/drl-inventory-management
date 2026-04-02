import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from project1_part_c import plot_training_curves, run_batch_settings, run_single_setting, save_results_json


REQUIRED_SETTINGS: List[Tuple[int, float]] = [(2, 4), (2, 9), (3, 4), (3, 9), (4, 4), (4, 9)]


PROFILE_CONFIGS: Dict[str, Dict] = {
    "baseline": {
        "dqn_kwargs": {},
        "ac_kwargs": {},
        "pg_kwargs": {},
        "dqn_train_kwargs": {"batch_size": 64, "warmup_steps": 200},
        "ac_train_kwargs": {},
        "pg_train_kwargs": {},
    },
    "tuned": {
        "dqn_kwargs": {
            "lr": 5e-4,
            "hidden_sizes": (256, 256),
            "epsilon_decay": 0.997,
            "target_update_freq": 100,
            "tau": 0.02,
            "double_q": True,
            "grad_clip": 5.0,
        },
        "ac_kwargs": {
            "actor_lr": 5e-4,
            "critic_lr": 1e-3,
            "hidden_sizes": (256, 256),
            "entropy_coef": 5e-3,
            "value_coef": 0.5,
        },
        "pg_kwargs": {
            "lr": 5e-4,
            "hidden_sizes": (256, 256),
            "baseline_momentum": 0.95,
            "entropy_coef": 5e-3,
        },
        "dqn_train_kwargs": {"batch_size": 128, "warmup_steps": 500},
        "ac_train_kwargs": {},
        "pg_train_kwargs": {},
    },
    "hybrid_dqn": {
        "dqn_kwargs": {
            "lr": 5e-4,
            "hidden_sizes": (256, 256),
            "epsilon_decay": 0.997,
            "target_update_freq": 100,
            "tau": 0.02,
            "double_q": True,
            "grad_clip": 5.0,
        },
        "ac_kwargs": {},
        "pg_kwargs": {},
        "dqn_train_kwargs": {"batch_size": 128, "warmup_steps": 500, "eval_horizon": 1000, "restore_best": True},
        "ac_train_kwargs": {},
        "pg_train_kwargs": {},
    },
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def flatten_result(result: Dict) -> List[Dict[str, object]]:
    config = result["config"]
    rows: List[Dict[str, object]] = []
    for name, payload in result["heuristics"].items():
        rows.append(
            {
                "l": config["l"],
                "p": config["p"],
                "method": name,
                "family": "heuristic",
                "avg_discounted_cost": payload["metrics"]["avg_discounted_cost"],
                "std": payload["metrics"]["avg_discounted_cost_std"],
                "details": json.dumps(payload["params"], ensure_ascii=True),
            }
        )
    for name, payload in result["drl"].items():
        rows.append(
            {
                "l": config["l"],
                "p": config["p"],
                "method": name,
                "family": "drl",
                "avg_discounted_cost": payload["final_metrics"]["avg_discounted_cost"],
                "std": payload["final_metrics"]["avg_discounted_cost_std"],
                "details": "DRL",
            }
        )
    return rows


def write_csv(rows: Iterable[Dict[str, object]], path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def best_cost(result: Dict, family: str) -> Tuple[str, float]:
    if family == "heuristic":
        name, payload = min(
            result["heuristics"].items(),
            key=lambda item: item[1]["metrics"]["avg_discounted_cost"],
        )
        return name, float(payload["metrics"]["avg_discounted_cost"])
    name, payload = min(
        result["drl"].items(),
        key=lambda item: item[1]["final_metrics"]["avg_discounted_cost"],
    )
    return name, float(payload["final_metrics"]["avg_discounted_cost"])


def write_markdown_summary(results: Sequence[Dict], path: Path, title: str) -> None:
    lines = [f"# {title}", "", "| l | p | best heuristic | heuristic cost | best DRL | DRL cost | gap |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for result in results:
        heuristic_name, heuristic_cost = best_cost(result, "heuristic")
        drl_name, drl_cost = best_cost(result, "drl")
        gap = drl_cost - heuristic_cost
        cfg = result["config"]
        lines.append(
            f"| {cfg['l']} | {cfg['p']} | {heuristic_name} | {heuristic_cost:.4f} | {drl_name} | {drl_cost:.4f} | {gap:.4f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_tuning(output_dir: Path, episodes: int, steps_per_episode: int, eval_runs: int, eval_every: int) -> None:
    ensure_dir(output_dir)
    heuristic_search_space = {"S_values": range(0, 31), "r_values": range(0, 16)}
    rows: List[Dict[str, object]] = []
    results: Dict[str, Dict] = {}
    for idx, profile_name in enumerate(("baseline", "tuned")):
        profile = PROFILE_CONFIGS[profile_name]
        result = run_single_setting(
            l=2,
            p=4,
            gamma=0.99,
            demand_lambda=5.0,
            max_order=30,
            heuristic_search_space=heuristic_search_space,
            episodes=episodes,
            steps_per_episode=steps_per_episode,
            eval_every=eval_every,
            eval_runs=eval_runs,
            seed=100 + idx,
            replay_capacity=20000,
            **profile,
        )
        results[profile_name] = result
        rows.extend([{**row, "profile": profile_name} for row in flatten_result(result)])
        plot_training_curves(result["drl"], output_dir=str(output_dir), prefix=f"tuning_{profile_name}_l2_p4")

    save_results_json(results, str(output_dir / "tuning_results.json"))
    write_csv(rows, output_dir / "tuning_results.csv")
    write_markdown_summary(list(results.values()), output_dir / "tuning_summary.md", "Tuning Summary")


def run_batch(output_dir: Path, episodes: int, steps_per_episode: int, eval_runs: int, eval_every: int) -> None:
    ensure_dir(output_dir)
    heuristic_search_space = {"S_values": range(0, 31), "r_values": range(0, 16)}
    profile = PROFILE_CONFIGS["hybrid_dqn"]
    results = run_batch_settings(
        settings=REQUIRED_SETTINGS,
        gamma=0.99,
        demand_lambda=5.0,
        max_order=30,
        heuristic_search_space=heuristic_search_space,
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        eval_every=eval_every,
        eval_runs=eval_runs,
        base_seed=200,
        replay_capacity=20000,
        **profile,
    )

    rows: List[Dict[str, object]] = []
    for result in results:
        cfg = result["config"]
        rows.extend(flatten_result(result))
        plot_training_curves(
            result["drl"],
            output_dir=str(output_dir),
            prefix=f"batch_l{cfg['l']}_p{int(cfg['p'])}",
        )

    save_results_json(results, str(output_dir / "batch_results.json"))
    write_csv(rows, output_dir / "batch_results.csv")
    write_markdown_summary(results, output_dir / "batch_summary.md", "Batch Summary")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tuning and batch experiments for the lost-sales project.")
    parser.add_argument("--mode", choices=("tune", "batch"), required=True)
    parser.add_argument("--output-dir", default="experiment_outputs")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--steps-per-episode", type=int, default=150)
    parser.add_argument("--eval-runs", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    ensure_dir(root)
    if args.mode == "tune":
        run_tuning(
            output_dir=root / "tuning",
            episodes=args.episodes,
            steps_per_episode=args.steps_per_episode,
            eval_runs=args.eval_runs,
            eval_every=args.eval_every,
        )
        return
    run_batch(
        output_dir=root / "batch",
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        eval_runs=args.eval_runs,
        eval_every=args.eval_every,
    )


if __name__ == "__main__":
    main()
