"""RMAPPO rollout collection, recurrent PPO update, checkpoint and evaluation."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch
from torch import Tensor

from .isaac_adapter import validate_backend_shapes
from .model import CentralizedCritic, SharedRecurrentActor
from .smoke_env import ContractSmokeEnv
from .storage import RolloutStorage, generalized_advantage_estimate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"configuration requests {requested}, but CUDA is unavailable; use --device cpu only for smoke tests"
        )
    return torch.device(requested)


def build_backend(cfg: Mapping[str, object], device: torch.device):
    backend_name = str(cfg["training"]["backend"])
    if backend_name == "smoke":
        backend = ContractSmokeEnv(cfg, device)
        validate_backend_shapes(backend)
        return backend
    if backend_name == "isaac":
        raise RuntimeError(
            "The high-fidelity Isaac backend is not wired yet. Do not fall back to the old single-drone "
            "LiDAR env.py: implement MultiUAVBackend in racer_rmappo/isaac_adapter.py first."
        )
    raise ValueError(f"unknown training backend: {backend_name}")


class RMAPPOTrainer:
    def __init__(self, cfg: Dict[str, object]) -> None:
        self.cfg = cfg
        seed = int(cfg["training"]["seed"])
        set_seed(seed)
        self.device = resolve_device(str(cfg["training"]["device"]))
        self.backend = build_backend(cfg, self.device)
        ppo = cfg["ppo"]
        depth = cfg["actor_observation"]["depth"]
        self.actor = SharedRecurrentActor(
            frame_stack=int(depth["frame_stack"]),
            hidden_size=int(ppo["recurrent_hidden_size"]),
        ).to(self.device)
        self.critic = CentralizedCritic(
            state_dim=13, hidden_size=int(ppo["recurrent_hidden_size"])
        ).to(self.device)
        parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=float(ppo["learning_rate"]), eps=1e-5)
        self.update_index = 0
        self.agent_transitions = 0
        output = Path(str(cfg["training"]["output_dir"]))
        if not output.is_absolute():
            output = Path(__file__).resolve().parents[3] / output
        self.output_dir = output
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metric_file = self.output_dir / "metrics.jsonl"

    def save(self, name: str) -> Path:
        path = self.output_dir / name
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "update_index": self.update_index,
                "agent_transitions": self.agent_transitions,
                "config": self.cfg,
            },
            path,
        )
        return path

    def load(self, path: str | Path, load_optimizer: bool = True) -> None:
        checkpoint = torch.load(Path(path).expanduser(), map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.update_index = int(checkpoint.get("update_index", 0))
        self.agent_transitions = int(checkpoint.get("agent_transitions", 0))

    def collect_rollout(self, observation, critic_state, hidden, episode_start):
        ppo = self.cfg["ppo"]
        storage = RolloutStorage()
        component_sums: Dict[str, float] = {}
        collision_sum = 0.0
        goal_sum = 0.0
        coverage_sum = 0.0
        for _ in range(int(ppo["rollout_steps"])):
            hidden_before = hidden
            with torch.no_grad():
                action, log_prob, _, next_hidden = self.actor.step(observation, hidden)
                value = self.critic(critic_state)
            next_observation, next_critic_state, reward, done, info = self.backend.step(action)
            done_agents = done.unsqueeze(-1).expand(-1, self.backend.num_agents)
            storage.add(
                observation,
                critic_state,
                action,
                log_prob,
                value,
                reward,
                done_agents,
                episode_start,
                hidden_before,
            )
            for key, component in info["reward_components"].items():
                component_sums[key] = component_sums.get(key, 0.0) + float(component.mean())
            collision_sum += float(info["collision"].mean())
            goal_sum += float(info["goal_reached"].mean())
            coverage_sum += float(info["coverage"].mean())
            hidden = next_hidden * (~done_agents).unsqueeze(-1)
            episode_start = done_agents
            observation = next_observation
            critic_state = next_critic_state

        with torch.no_grad():
            bootstrap_value = self.critic(critic_state)
        batch = storage.stack()
        advantages, returns = generalized_advantage_estimate(
            batch["rewards"],
            batch["values"],
            batch["dones"],
            bootstrap_value,
            float(ppo["gamma"]),
            float(ppo["gae_lambda"]),
        )
        batch["advantages"] = advantages
        batch["returns"] = returns
        steps = int(ppo["rollout_steps"])
        rollout_metrics = {
            f"reward/{key}": value / steps for key, value in component_sums.items()
        }
        rollout_metrics.update(
            {
                "env/collision_team_rate": collision_sum / steps,
                "env/goals_per_swarm_step": goal_sum / steps,
                "env/coverage_mean": coverage_sum / steps,
                "rollout/reward_mean": float(batch["rewards"].mean()),
            }
        )
        return batch, observation, critic_state, hidden, episode_start, rollout_metrics

    def update(self, batch: Dict[str, object]) -> Dict[str, float]:
        ppo = self.cfg["ppo"]
        advantages: Tensor = batch["advantages"]
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
        old_log_probs: Tensor = batch["log_probs"]
        old_values: Tensor = batch["values"]
        returns: Tensor = batch["returns"]
        rollout_steps, env_count = old_log_probs.shape[:2]
        sequence_length = int(ppo["recurrent_sequence_length"])
        sequence_starts = list(range(0, rollout_steps, sequence_length))
        sequence_index = [
            (start, env_id) for start in sequence_starts for env_id in range(env_count)
        ]
        loss_totals: Dict[str, float] = {}
        update_count = 0

        for _ in range(int(ppo["epochs"])):
            permutation = torch.randperm(len(sequence_index), device=self.device).tolist()
            minibatch_count = min(int(ppo["minibatches"]), len(sequence_index))
            for permutation_chunk in torch.tensor_split(
                torch.tensor(permutation, device=self.device), minibatch_count
            ):
                selected = [sequence_index[int(index)] for index in permutation_chunk]
                def sequence_batch(value: Tensor) -> Tensor:
                    return torch.stack(
                        [value[start : start + sequence_length, env_id] for start, env_id in selected],
                        dim=1,
                    )

                observations = {
                    key: sequence_batch(value) for key, value in batch["observations"].items()
                }
                action_batch = sequence_batch(batch["actions"])
                start_batch = sequence_batch(batch["episode_starts"])
                hidden = torch.stack(
                    [batch["hidden_states"][start, env_id] for start, env_id in selected], dim=0
                )
                new_log_prob, entropy = self.actor.evaluate_sequence(
                    observations, hidden, action_batch, start_batch
                )
                values = self.critic(sequence_batch(batch["critic_states"]))
                old_log = sequence_batch(old_log_probs)
                advantage = sequence_batch(advantages)
                ratio = torch.exp(new_log_prob - old_log)
                unclipped = ratio * advantage
                clipped = ratio.clamp(
                    1.0 - float(ppo["clip_ratio"]), 1.0 + float(ppo["clip_ratio"])
                ) * advantage
                policy_loss = -torch.minimum(unclipped, clipped).mean()

                previous_values = sequence_batch(old_values)
                return_batch = sequence_batch(returns)
                clipped_values = previous_values + (values - previous_values).clamp(
                    -float(ppo["clip_ratio"]), float(ppo["clip_ratio"])
                )
                value_loss = 0.5 * torch.maximum(
                    (values - return_batch).square(), (clipped_values - return_batch).square()
                ).mean()
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + float(ppo["value_coefficient"]) * value_loss
                    - float(ppo["entropy_coefficient"]) * entropy_mean
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    float(ppo["max_grad_norm"]),
                )
                self.optimizer.step()

                approximate_kl = ((ratio - 1.0) - (new_log_prob - old_log)).mean().detach()
                clip_fraction = ((ratio - 1.0).abs() > float(ppo["clip_ratio"])).float().mean()
                values_to_add = {
                    "loss/total": float(loss.detach()),
                    "loss/policy": float(policy_loss.detach()),
                    "loss/value": float(value_loss.detach()),
                    "policy/entropy": float(entropy_mean.detach()),
                    "policy/approx_kl": float(approximate_kl),
                    "policy/clip_fraction": float(clip_fraction),
                    "policy/gradient_norm": float(gradient_norm),
                }
                for key, value in values_to_add.items():
                    loss_totals[key] = loss_totals.get(key, 0.0) + value
                update_count += 1

        return {key: value / max(update_count, 1) for key, value in loss_totals.items()}

    def train(self, total_environment_steps: int | None = None) -> Path:
        observation, critic_state = self.backend.reset()
        hidden = self.actor.initial_hidden(
            self.backend.num_envs, self.backend.num_agents, self.device
        )
        episode_start = torch.ones(
            self.backend.num_envs, self.backend.num_agents, dtype=torch.bool, device=self.device
        )
        ppo = self.cfg["ppo"]
        transitions_per_update = (
            int(ppo["rollout_steps"]) * self.backend.num_envs * self.backend.num_agents
        )
        target = int(total_environment_steps or self.cfg["training"]["total_environment_steps"])
        remaining_updates = max(1, math.ceil(max(target - self.agent_transitions, 1) / transitions_per_update))
        start_time = time.monotonic()
        for _ in range(remaining_updates):
            batch, observation, critic_state, hidden, episode_start, metrics = self.collect_rollout(
                observation, critic_state, hidden, episode_start
            )
            metrics.update(self.update(batch))
            self.update_index += 1
            self.agent_transitions += transitions_per_update
            metrics.update(
                {
                    "update": self.update_index,
                    "agent_transitions": self.agent_transitions,
                    "elapsed_seconds": time.monotonic() - start_time,
                }
            )
            with self.metric_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics, sort_keys=True) + "\n")
            if self.update_index % int(self.cfg["training"]["log_interval_updates"]) == 0:
                print(
                    f"update={self.update_index} transitions={self.agent_transitions} "
                    f"reward={metrics['rollout/reward_mean']:.4f} "
                    f"collision={metrics['env/collision_team_rate']:.4f} "
                    f"kl={metrics['policy/approx_kl']:.5f}"
                )
            if self.update_index % int(self.cfg["training"]["checkpoint_interval_updates"]) == 0:
                self.save(f"checkpoint_{self.update_index:06d}.pt")
        return self.save("checkpoint_final.pt")

    @torch.no_grad()
    def evaluate(self, steps: int = 1024) -> Dict[str, float]:
        observation, critic_state = self.backend.reset()
        hidden = self.actor.initial_hidden(
            self.backend.num_envs, self.backend.num_agents, self.device
        )
        reward_sum = 0.0
        collisions = 0.0
        goals = 0.0
        coverage = 0.0
        for _ in range(steps):
            action, _, _, next_hidden = self.actor.step(observation, hidden, deterministic=True)
            observation, critic_state, reward, done, info = self.backend.step(action)
            done_agents = done.unsqueeze(-1).expand(-1, self.backend.num_agents)
            hidden = next_hidden * (~done_agents).unsqueeze(-1)
            reward_sum += float(reward.mean())
            collisions += float(info["collision"].mean())
            goals += float(info["goal_reached"].mean())
            coverage += float(info["coverage"].mean())
        return {
            "reward_mean": reward_sum / steps,
            "collision_team_rate": collisions / steps,
            "goals_per_swarm_step": goals / steps,
            "coverage_mean": coverage / steps,
        }
