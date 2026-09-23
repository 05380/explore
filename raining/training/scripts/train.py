"""Training entrypoint: collect exploratory rollouts, update PPO, and periodically run deterministic evaluation."""

import argparse
import os
import hydra
import datetime
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from omni.isaac.kit import SimulationApp
from ppo import PPO
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController, ravel_composite
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
from torchrl.envs.transforms import TransformedEnv, Compose
from utils import evaluate, resolve_eval_style, summarize_episode_stats
from torchrl.envs.utils import ExplorationType
from demonstration_buffer import DemonstrationBuffer
from wall_follow_teacher import TeacherRolloutPolicy, WallFollowTeacher




FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")


def print_eval_metrics(eval_info, prefix="eval", label="eval"):
    metric_keys = [
        f"{prefix}/success_rate",
        f"{prefix}/collision_rate",
        f"{prefix}/wall_collision_rate",
        f"{prefix}/below_bound_rate",
        f"{prefix}/above_bound_rate",
        f"{prefix}/horizontal_out_of_bound_rate",
        f"{prefix}/deadlock_rate",
        f"{prefix}/time_limit_rate",
        f"{prefix}/stall_rate",
        f"{prefix}/stats.return",
        f"{prefix}/stats.episode_len",
        f"{prefix}/stats.reward_goal_progress",
        f"{prefix}/stats.penalty_safety_static",
        f"{prefix}/stats.penalty_safety_dynamic",
        f"{prefix}/stats.reward_vel",
        f"{prefix}/stats.penalty_height",
        f"{prefix}/stats.reward_escape",
        f"{prefix}/vo_risk_mean",
    ]
    metric_parts = []
    for key in metric_keys:
        value = eval_info.get(key)
        if value is not None:
            metric_parts.append(f"{key}={value:.4f}")
    if metric_parts:
        print(f"[NavRL]: {label} metrics | " + " | ".join(metric_parts))


@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    # Simulation App
    #sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1, "extra_args": ["--/persistent/isaac/asset_root/default=http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.0"]})

    # Use Wandb to monitor training
    if (cfg.wandb.run_id is None):
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            config=cfg,
            mode=cfg.wandb.mode,
            id=wandb.util.generate_id(),
        )
    else:
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            config=cfg,
            mode=cfg.wandb.mode,
            id=cfg.wandb.run_id,
            resume="must"
        )

    run.define_metric("env_frames")
    run.define_metric("ppo_env_frames")
    run.define_metric("eval/*", step_metric="env_frames")
    run.define_metric("teacher/*", step_metric="env_frames")
    for metric_name in [
        "train/stall_reward_mean",
        "train/stall/reward_mean",
        "train/stall/active_rate",
        "train/stall_rate",
        "train/stall_steps",
        "train/stall/step_ratio",
        "train/vo_reward_mean",
        "train/vo/reward_mean",
        "train/vo/active_rate",
        "train/vo_risk_mean",
        "train/vo/risk_mean",
        "train/vo/risk_active_rate",
        "train/vo_warmup",
        "train/vo/warmup",
        "train/below_bound_rate",
        "train/above_bound_rate",
        "train/horizontal_out_of_bound_rate",
        "train/wall_collision_rate",
        "eval/wall_collision_rate",
        "eval/stall_reward_mean",
        "eval/stall/reward_mean",
        "eval/stall/active_rate",
        "eval/stall_rate",
        "eval/stall_steps",
        "eval/stall/step_ratio",
        "eval/vo_reward_mean",
        "eval/vo/reward_mean",
        "eval/vo/active_rate",
        "eval/vo_risk_mean",
        "eval/vo/risk_mean",
        "eval/vo/risk_active_rate",
        "eval/vo_warmup",
        "eval/vo/warmup",
        "eval/below_bound_rate",
        "eval/above_bound_rate",
        "eval/horizontal_out_of_bound_rate",
        "train/stats.reward_goal_progress",
        "train/stats.penalty_safety_static",
        "train/stats.penalty_safety_dynamic",
        "train/stats.reward_vel",
        "train/stats.penalty_height",
        "train/stats.reward_escape",
        "train/stats.reward_stall",
        "train/stats.stall",
        "train/stats.stall_steps",
        "train/stats.reward_vo",
        "train/stats.vo_risk",
        "train/stats.vo_warmup",
        "train/stats.below_bound",
        "train/stats.above_bound",
        "train/stats.horizontal_out_of_bound",
        "train/stats.wall_collision",
        "eval/stats.reward_stall",
        "eval/stats.stall",
        "eval/stats.stall_steps",
        "eval/stats.reward_goal_progress",
        "eval/stats.penalty_safety_static",
        "eval/stats.penalty_safety_dynamic",
        "eval/stats.reward_vel",
        "eval/stats.penalty_height",
        "eval/stats.reward_escape",
        "eval/stats.reward_vo",
        "eval/stats.vo_risk",
        "eval/stats.vo_warmup",
        "eval/stats.below_bound",
        "eval/stats.above_bound",
        "eval/stats.horizontal_out_of_bound",
        "eval/stats.wall_collision",
        "auxiliary_loss",
        "auxiliary_future_progress_loss",
        "auxiliary_future_clearance_loss",
        "auxiliary_future_collision_loss",
        "auxiliary_future_stuck_loss",
        "behavior_cloning_loss",
        "behavior_cloning_nll",
        "behavior_cloning_confidence",
        "behavior_cloning_active",
        "feature_grad_norm",
    ]:#
        run.define_metric(metric_name, step_metric="env_frames")

    # Navigation Training Environment
    from env import NavigationEnv
    env = NavigationEnv(cfg)

    # Transformed Environment
    transforms = []
    # transforms.append(ravel_composite(env.observation_spec, ("agents", "intrinsics"), start_dim=-1))
    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=False)
    transforms.append(vel_transform)
    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    transformed_env.set_seed(cfg.seed)    
    # PPO Policy
    policy = PPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)

    if cfg.get("checkpoint", None) is not None:
        checkpoint_path = os.path.expanduser(str(cfg.checkpoint))
        checkpoint_state = torch.load(checkpoint_path, map_location=cfg.device)
        missing_keys, unexpected_keys = policy.load_state_dict(checkpoint_state, strict=False)
        if missing_keys or unexpected_keys:
            print("[NavRL]: checkpoint loaded with non-strict key match.")
            if missing_keys:
                print("[NavRL]: missing keys:", missing_keys)
            if unexpected_keys:
                print("[NavRL]: unexpected keys:", unexpected_keys)
        print("[NavRL]: loaded checkpoint from: ", checkpoint_path)

    collector_policy = policy
    demonstration_buffer = None
    teacher_rollout_policy = None
    teacher_env_count = 0
    teacher_cfg = cfg.get("wall_teacher", {})
    behavior_cloning_cfg = cfg.algo.feature_extractor.get("behavior_cloning", {})
    teacher_enabled = bool(teacher_cfg.get("enabled", False))
    behavior_cloning_enabled = bool(behavior_cloning_cfg.get("enabled", False))
    if teacher_enabled and behavior_cloning_enabled and int(cfg.env.num_envs) > 1:
        requested_count = int(teacher_cfg.get("num_envs", 0))
        if requested_count <= 0:
            requested_count = max(
                int(round(float(cfg.env.num_envs) * float(teacher_cfg.get("env_fraction", 0.0)))),
                1,
            )
        maximum_count = max(int(teacher_cfg.get("max_envs", requested_count)), 1)
        teacher_env_count = min(requested_count, maximum_count, int(cfg.env.num_envs) - 1)

        teacher_env_mask = torch.zeros(
            int(cfg.env.num_envs),
            dtype=torch.bool,
            device=cfg.device,
        )
        teacher_env_mask[:teacher_env_count] = True
        demonstration_buffer = DemonstrationBuffer(
            capacity=int(behavior_cloning_cfg.get("buffer_capacity", 100000)),
            balanced_sampling=bool(behavior_cloning_cfg.get("balanced_sampling", True)),
            phase_balance_exponent=float(
                behavior_cloning_cfg.get("phase_balance_exponent", 0.5)
            ),
        )
        wall_teacher = WallFollowTeacher(
            cfg=teacher_cfg,
            sensor_cfg=cfg.sensor,
            vo_cfg=cfg.reward.vo,
            num_envs=int(cfg.env.num_envs),
            teacher_env_mask=teacher_env_mask,
            action_limit=float(cfg.algo.actor.action_limit),
            dt=float(env.dt),
            device=cfg.device,
        )
        teacher_rollout_policy = TeacherRolloutPolicy(
            base_policy=policy,
            teacher=wall_teacher,
            demonstration_buffer=demonstration_buffer,
        )
        collector_policy = teacher_rollout_policy
        print(
            f"[NavRL]: wall teacher enabled in {teacher_env_count} environments; "
            f"{int(cfg.env.num_envs) - teacher_env_count} environments remain on-policy."
        )
    elif teacher_enabled and not behavior_cloning_enabled:
        print("[NavRL]: wall teacher disabled because behavior cloning is disabled.")
    elif teacher_enabled and int(cfg.env.num_envs) <= 1:
        print("[NavRL]: wall teacher disabled because at least one on-policy environment is required.")

    # Episode Stats Collector
    episode_stats_keys = [
        k for k in transformed_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(episode_stats_keys)

    # RL Data Collector
    collector = SyncDataCollector(
        transformed_env,
        policy=collector_policy,
        frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num, 
        total_frames=cfg.max_frame_num,
        device=cfg.device,
        return_same_td=True, # update the return tensordict inplace (should set to false if we need to use replace buffer)
        exploration_type=ExplorationType.RANDOM, # sample from normal distribution
    )
    if teacher_env_count > 0:
        if not hasattr(collector.policy, "demonstration_buffer"):
            raise RuntimeError("Collector policy lost the wall-teacher demonstration buffer.")
        teacher_rollout_policy = collector.policy
        demonstration_buffer = collector.policy.demonstration_buffer

    # Training Loop
    for i, data in enumerate(collector):
        # print("data: ", data)
        # print("============================")
        # Log Info
        if not data.batch_size or int(data.batch_size[0]) != int(cfg.env.num_envs):
            raise RuntimeError(
                "Wall-teacher isolation requires collector batches ordered as "
                "[num_envs, rollout_steps, ...]."
            )
        ppo_data = data[teacher_env_count:] if teacher_env_count > 0 else data
        ppo_env_count = int(cfg.env.num_envs) - teacher_env_count
        info = {
            "env_frames": collector._frames,
            "ppo_env_frames": (i + 1) * ppo_env_count * int(cfg.algo.training_frame_num),
            "rollout_fps": collector._fps,
        }

        # Train Policy
        train_loss_stats = policy.train(ppo_data, demonstration_buffer)
        collector.update_policy_weights_()
        info.update(train_loss_stats) # log training loss info
        if teacher_rollout_policy is not None:
            info.update(teacher_rollout_policy.pop_metrics())

        # Calculate and log training episode stats
        episode_stats.add(ppo_data)
        if len(episode_stats) >= ppo_env_count: # evaluate once if all on-policy agents finished one episode
            stats = summarize_episode_stats(episode_stats.pop(), prefix="train")
            info.update(stats)
            train_metric_parts = []
            for key in ["train/collision_rate", "train/wall_collision_rate"]:
                value = stats.get(key)
                if value is not None:
                    train_metric_parts.append(f"{key}={value:.4f}")
            if train_metric_parts and i % cfg.eval_interval == 0:
                print("[NavRL]: train metrics | " + " | ".join(train_metric_parts))

        # Evaluate policy and log info
        if i % cfg.eval_interval == 0:
            print("[NavRL]: start evaluating policy at training step: ", i)
            eval_task_mode, eval_label = resolve_eval_style(cfg)
            eval_info = evaluate(
                env=transformed_env, 
                policy=policy,
                seed=cfg.seed, 
                cfg=cfg,
                exploration_type=ExplorationType.MEAN,
                prefix="eval",
                eval_task_mode=eval_task_mode,
            )
            info.update(eval_info)
            print_eval_metrics(eval_info, prefix="eval", label=eval_label)
            print("\n[NavRL]: evaluation done.")
        
        # Update wand info
        run.log(info)


        # Save Model
        if i % cfg.save_interval == 0:
            ckpt_path = os.path.join(run.dir, f"checkpoint_{i}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            print("[NavRL]: model saved at training step: ", i)

    ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
    torch.save(policy.state_dict(), ckpt_path)
    wandb.finish()
    sim_app.close()

if __name__ == "__main__":
    main()
    
