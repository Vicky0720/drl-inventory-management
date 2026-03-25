from pprint import pprint

from project1_part_a import Lostsale
from project1_part_b import AC_agent, DQN_agent, Replaybuffer
from project1_part_c import PG_agent, run_single_setting, summarize_results


def main() -> None:
    env = Lostsale(buffer_size=5, gamma=0.99, p=4, l=2, max_order=8, seed=21)
    state_dim = env.lead + 1
    action_dim = env.max_order + 1

    dqn_agent = DQN_agent(state_dim=state_dim, action_dim=action_dim, gamma=env.discount, target_update_freq=10)
    ac_agent = AC_agent(state_dim=state_dim, action_dim=action_dim, gamma=env.discount)
    pg_agent = PG_agent(state_dim=state_dim, action_dim=action_dim, gamma=env.discount)
    replay_buffer = Replaybuffer(capacity=1000, state_dim=state_dim)

    result = run_single_setting(
        l=2,
        p=4,
        dqn_agent=dqn_agent,
        ac_agent=ac_agent,
        pg_agent=pg_agent,
        replay_buffer=replay_buffer,
        max_order=8,
        episodes=6,
        steps_per_episode=20,
        eval_every=3,
        eval_runs=3,
        heuristic_search_space={
            "S_values": range(0, 10),
            "r_values": range(0, 9),
        },
        seed=21,
    )

    print("Config:")
    pprint(result["config"])
    print("\nMethods summary:")
    for row in summarize_results(result):
        print(row)


if __name__ == "__main__":
    main()
