from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from project1_part_a import Lostsale, average_discounted_cost


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(x, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=DEVICE)


class Replaybuffer:
    """Simple replay buffer for value-based DRL."""

    def __init__(
        self,
        capacity: int = 10000,
        state_dim: Optional[int] = None,
    ) -> None:
        self.capacity = int(capacity)
        self.state_dim = state_dim
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, float]] = []
        self.position = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        state: Sequence[float],
        action: int,
        reward: float,
        next_state: Sequence[float],
        done: bool = False,
    ) -> None:
        transition = (
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[i] for i in indices))
        return {
            "states": to_tensor(np.stack(states)),
            "actions": to_tensor(actions, dtype=torch.long),
            "rewards": to_tensor(rewards),
            "next_states": to_tensor(np.stack(next_states)),
            "dones": to_tensor(dones),
        }


class QNetwork(nn.Module):
    """Feedforward Q-network for discrete order quantities."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = state_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Policy(nn.Module):
    """Categorical policy network for discrete order quantities."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = state_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def distribution(self, x: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.forward(x)
        return torch.distributions.Categorical(logits=logits)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_sizes: Sequence[int] = (128, 128)) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = state_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class TrainLogs:
    losses: List[float]
    eval_costs: List[float]


class DQN_agent:
    """Discrete-action DQN for inventory replenishment."""

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
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        self.q_net = QNetwork(state_dim, action_dim, hidden_sizes).to(DEVICE)
        self.target_q_net = QNetwork(state_dim, action_dim, hidden_sizes).to(DEVICE)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.train_steps = 0

    def select_action(self, state: Sequence[float], greedy: bool = False) -> int:
        if (not greedy) and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))
        state_t = to_tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def update(self, batch: Dict[str, torch.Tensor]) -> float:
        q_values = self.q_net(batch["states"])
        current_q = q_values.gather(1, batch["actions"].unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_q_net(batch["next_states"]).max(dim=1).values
            target_q = batch["rewards"] + self.gamma * (1.0 - batch["dones"]) * next_q

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(loss.item())

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
    ) -> TrainLogs:
        losses: List[float] = []
        eval_costs: List[float] = []

        total_steps = 0
        for episode in range(episodes):
            state = env.reset()
            for _ in range(steps_per_episode):
                action = self.select_action(state)
                next_state, cost = env.step(state, action)
                reward = -cost
                replay_buffer.push(state, action, reward, next_state, False)
                state = next_state
                total_steps += 1

                if len(replay_buffer) >= max(batch_size, warmup_steps):
                    batch = replay_buffer.sample(batch_size)
                    loss = self.update(batch)
                    losses.append(loss)

            if (episode + 1) % eval_every == 0:
                metrics = self.evaluate(env, horizon=steps_per_episode, n_sim=eval_runs)
                eval_costs.append(metrics["avg_discounted_cost"])

        return TrainLogs(losses=losses, eval_costs=eval_costs)

    def evaluate(self, env: Lostsale, horizon: int = 1000, n_sim: int = 100) -> Dict[str, float]:
        from project1_part_a import evaluate_policy

        return evaluate_policy(env, lambda state: self.select_action(state, greedy=True), horizon=horizon, n_sim=n_sim)


class AC_agent:
    """Discrete Actor-Critic using a categorical actor and a state-value critic."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        hidden_sizes: Sequence[int] = (128, 128),
        entropy_coef: float = 1e-3,
        value_coef: float = 1.0,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.actor = Policy(state_dim, action_dim, hidden_sizes).to(DEVICE)
        self.critic = ValueNetwork(state_dim, hidden_sizes).to(DEVICE)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state: Sequence[float], greedy: bool = False) -> int:
        state_t = to_tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor.distribution(state_t)
            if greedy:
                action = torch.argmax(dist.logits, dim=1)
            else:
                action = dist.sample()
        return int(action.item())

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> Dict[str, float]:
        states_t = to_tensor(states)
        actions_t = to_tensor(actions, dtype=torch.long)
        rewards_t = to_tensor(rewards)
        next_states_t = to_tensor(next_states)
        dones_t = to_tensor(dones)

        dist = self.actor.distribution(states_t)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        values = self.critic(states_t)
        with torch.no_grad():
            next_values = self.critic(next_states_t)
            targets = rewards_t + self.gamma * (1.0 - dones_t) * next_values
            advantages = targets - values

        policy_loss = -(log_probs * advantages.detach()).mean() - self.entropy_coef * entropy
        value_loss = F.mse_loss(values, targets)

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        (self.value_coef * value_loss).backward()
        self.critic_optimizer.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
        }

    def train(
        self,
        env: Lostsale,
        episodes: int = 200,
        steps_per_episode: int = 100,
        eval_every: int = 20,
        eval_runs: int = 20,
    ) -> Dict[str, List[float]]:
        policy_losses: List[float] = []
        value_losses: List[float] = []
        eval_costs: List[float] = []

        for episode in range(episodes):
            state = env.reset()
            states: List[np.ndarray] = []
            actions: List[int] = []
            rewards: List[float] = []
            next_states: List[np.ndarray] = []
            dones: List[float] = []

            for _ in range(steps_per_episode):
                action = self.select_action(state)
                next_state, cost = env.step(state, action)
                reward = -cost

                states.append(np.asarray(state, dtype=np.float32))
                actions.append(action)
                rewards.append(reward)
                next_states.append(np.asarray(next_state, dtype=np.float32))
                dones.append(0.0)
                state = next_state

            logs = self.update(
                np.stack(states),
                np.asarray(actions, dtype=np.int64),
                np.asarray(rewards, dtype=np.float32),
                np.stack(next_states),
                np.asarray(dones, dtype=np.float32),
            )
            policy_losses.append(logs["policy_loss"])
            value_losses.append(logs["value_loss"])

            if (episode + 1) % eval_every == 0:
                metrics = self.evaluate(env, horizon=steps_per_episode, n_sim=eval_runs)
                eval_costs.append(metrics["avg_discounted_cost"])

        return {
            "policy_losses": policy_losses,
            "value_losses": value_losses,
            "eval_costs": eval_costs,
        }

    def evaluate(self, env: Lostsale, horizon: int = 1000, n_sim: int = 100) -> Dict[str, float]:
        from project1_part_a import evaluate_policy

        return evaluate_policy(env, lambda state: self.select_action(state, greedy=True), horizon=horizon, n_sim=n_sim)

