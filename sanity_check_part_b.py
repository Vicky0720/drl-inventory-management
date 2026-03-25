from pprint import pprint

from project1_part_a import Lostsale
from project1_part_b import AC_agent, DQN_agent, Replaybuffer


def main() -> None:
    env = Lostsale(buffer_size=5, gamma=0.99, p=4, l=2, max_order=10, seed=11)
    state_dim = env.lead + 1
    action_dim = env.max_order + 1

    replay_buffer = Replaybuffer(capacity=1000, state_dim=state_dim)
    dqn_agent = DQN_agent(state_dim=state_dim, action_dim=action_dim, target_update_freq=10)
    ac_agent = AC_agent(state_dim=state_dim, action_dim=action_dim)

    dqn_logs = dqn_agent.train(
        env,
        replay_buffer,
        episodes=6,
        steps_per_episode=20,
        batch_size=16,
        warmup_steps=16,
        eval_every=3,
        eval_runs=3,
    )
    print("DQN losses logged:", len(dqn_logs.losses))
    print("DQN eval costs:", dqn_logs.eval_costs)

    ac_logs = ac_agent.train(
        env,
        episodes=6,
        steps_per_episode=20,
        eval_every=3,
        eval_runs=3,
    )
    print("AC policy losses:", len(ac_logs["policy_losses"]))
    print("AC value losses:", len(ac_logs["value_losses"]))
    print("AC eval costs:", ac_logs["eval_costs"])

    dqn_eval = dqn_agent.evaluate(env, horizon=50, n_sim=3)
    ac_eval = ac_agent.evaluate(env, horizon=50, n_sim=3)
    print("\nDQN eval metrics:")
    pprint(dqn_eval)
    print("\nAC eval metrics:")
    pprint(ac_eval)


if __name__ == "__main__":
    main()
