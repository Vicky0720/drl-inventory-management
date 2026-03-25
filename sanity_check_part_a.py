from pprint import pprint

import numpy as np

from project1_part_a import (
    Basestock,
    Constantorder,
    Lostsale,
    evaluate_policy,
    run_heuristic_benchmarks,
)


def main() -> None:
    env = Lostsale(buffer_size=5, gamma=0.99, p=4, l=2, max_order=15, seed=7)

    state = env.reset([3, 2, 1])
    next_state, cost = env.step(state, action=4, demand=6)
    print("Transition check:")
    print("state     =", state)
    print("next_state=", next_state)
    print("cost      =", cost)

    assert np.allclose(next_state, np.array([0.0, 1.0, 4.0]))
    assert abs(cost - 4.0) < 1e-9

    constant_metrics = evaluate_policy(env, lambda s: Constantorder(r=5), horizon=50, n_sim=5)
    print("\nConstant-order metrics:")
    pprint(constant_metrics)

    base_action = Basestock(state=np.array([3.0, 2.0, 1.0]), S=8, l=2)
    print("\nBase-stock action check:", base_action)
    assert base_action == 4

    benchmark_results = run_heuristic_benchmarks(
        env,
        S_values=range(0, 16),
        r_values=range(0, 11),
        horizon=100,
        n_sim=10,
        max_order=15,
    )
    print("\nHeuristic benchmark summary:")
    for name, result in benchmark_results.items():
        print(name, result["params"], result["metrics"]["avg_discounted_cost"])


if __name__ == "__main__":
    main()
