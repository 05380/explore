import torch
import torch.nn as nn
import wandb
import numpy as np
from typing import Iterable, Union
from tensordict.tensordict import TensorDict
from omni_drones.utils.torchrl import RenderCallback
from torchrl.envs.utils import ExplorationType, set_exploration_type

class ValueNorm(nn.Module):
    def __init__(
        self,
        input_shape: Union[int, Iterable],
        beta=0.995,
        epsilon=1e-5,
    ) -> None:
        super().__init__()

        self.input_shape = (
            torch.Size(input_shape)
            if isinstance(input_shape, Iterable)
            else torch.Size((input_shape,))
        )
        self.epsilon = epsilon
        self.beta = beta

        self.running_mean: torch.Tensor
        self.running_mean_sq: torch.Tensor
        self.debiasing_term: torch.Tensor
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_mean_sq", torch.zeros(input_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(
            min=self.epsilon
        )
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        dim = tuple(range(input_vector.dim() - len(self.input_shape)))
        batch_mean = input_vector.mean(dim=dim)
        batch_sq_mean = (input_vector**2).mean(dim=dim)

        weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = (input_vector - mean) / torch.sqrt(var)
        return out

    def denormalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var) + mean
        return out

def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)

class IndependentNormal(torch.distributions.Independent):
    arg_constraints = {"loc": torch.distributions.constraints.real, "scale": torch.distributions.constraints.positive} 
    def __init__(self, loc, scale, validate_args=None):
        scale = torch.clamp_min(scale, 1e-6)
        base_dist = torch.distributions.Normal(loc, scale)
        super().__init__(base_dist, 1, validate_args=validate_args)

class IndependentBeta(torch.distributions.Independent):
    arg_constraints = {"alpha": torch.distributions.constraints.positive, "beta": torch.distributions.constraints.positive}

    def __init__(self, alpha, beta, validate_args=None):
        beta_dist = torch.distributions.Beta(alpha, beta)
        super().__init__(beta_dist, 1, validate_args=validate_args)

class Actor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.zeros(action_dim)) 
    
    def forward(self, features: torch.Tensor):
        loc = self.actor_mean(features)
        scale = torch.exp(self.actor_std).expand_as(loc)
        return loc, scale

class BetaActor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.alpha_layer = nn.LazyLinear(action_dim)
        self.beta_layer = nn.LazyLinear(action_dim)
        self.alpha_softplus = nn.Softplus()
        self.beta_softplus = nn.Softplus()
    
    def forward(self, features: torch.Tensor):
        alpha = 1. + self.alpha_softplus(self.alpha_layer(features)) + 1e-6
        beta = 1. + self.beta_softplus(self.beta_layer(features)) + 1e-6
        # print("alpha: ", alpha)
        # print("beta: ", beta)
        return alpha, beta

class GAE(nn.Module):
    def __init__(self, gamma, lmbda):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))
        self.gamma: torch.Tensor
        self.lmbda: torch.Tensor
    
    def forward(
        self, 
        reward: torch.Tensor, 
        terminated: torch.Tensor, 
        value: torch.Tensor, 
        next_value: torch.Tensor
    ):
        num_steps = terminated.shape[1]
        advantages = torch.zeros_like(reward)
        not_done = 1 - terminated.float()
        gae = 0
        for step in reversed(range(num_steps)):
            delta = (
                reward[:, step] 
                + self.gamma * next_value[:, step] * not_done[:, step] 
                - value[:, step]
            )
            advantages[:, step] = gae = delta + (self.gamma * self.lmbda * not_done[:, step] * gae) 
        returns = advantages + value
        return advantages, returns

def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1) 
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]


def summarize_episode_stats(stats, prefix: str):
    try:
        stats_items = stats.items(True, True)
    except TypeError:
        stats_items = stats.items()

    flat_stats = {}
    for key, value in stats_items:
        if isinstance(key, tuple):
            key_parts = key[1:] if len(key) > 0 and key[0] == "stats" else key
            name = ".".join(key_parts)
        else:
            name = key
        flat_stats[name] = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)

    info = {
        f"{prefix}/stats.{k}": torch.mean(v.float()).item()
        for k, v in flat_stats.items()
    }

    def mean_rate(mask: torch.Tensor):
        return mask.float().mean().item()

    def conditional_rate(event: torch.Tensor, condition: torch.Tensor):
        condition = condition.bool()
        return event[condition].float().mean().item() if condition.any() else 0.0

    reach_goal = flat_stats.get("reach_goal")
    collision = flat_stats.get("collision")
    wall_collision = flat_stats.get("wall_collision")
    below_bound = flat_stats.get("below_bound")
    above_bound = flat_stats.get("above_bound")
    horizontal_out_of_bound = flat_stats.get("horizontal_out_of_bound")
    truncated = flat_stats.get("truncated")
    deadlock = flat_stats.get("stuck")
    stuck_steps = flat_stats.get("stuck_steps")
    episode_len = flat_stats.get("episode_len")
    stall = flat_stats.get("stall")
    stall_steps = flat_stats.get("stall_steps")
    reward_stall = flat_stats.get("reward_stall")
    reward_vo = flat_stats.get("reward_vo")
    vo_risk = flat_stats.get("vo_risk")
    vo_warmup = flat_stats.get("vo_warmup")

    success_mask = None
    failure_mask = None
    collision_mask = None
    wall_collision_mask = None
    time_limit_mask = None
    deadlock_mask = None

    if reach_goal is not None:
        success_mask = reach_goal >= 0.5
        failure_mask = ~success_mask
        success_rate = mean_rate(success_mask)
        failure_rate = mean_rate(failure_mask)
        info[f"{prefix}/success_rate"] = success_rate
        info[f"{prefix}/failure_rate"] = failure_rate
        info[f"{prefix}/rates/success"] = success_rate
        info[f"{prefix}/rates/failure"] = failure_rate

    if collision is not None:
        collision_mask = collision >= 0.5
        collision_rate = mean_rate(collision_mask)
        info[f"{prefix}/collision_rate"] = collision_rate
        info[f"{prefix}/rates/collision"] = collision_rate

    if wall_collision is not None:
        wall_collision_mask = wall_collision >= 0.5
        wall_collision_rate = mean_rate(wall_collision_mask)
        info[f"{prefix}/wall_collision_rate"] = wall_collision_rate
        info[f"{prefix}/rates/wall_collision"] = wall_collision_rate
        if collision_mask is not None:
            info[f"{prefix}/conditioned/wall_collision_given_collision"] = conditional_rate(
                wall_collision_mask, collision_mask
            )

    if below_bound is not None:
        below_bound_mask = below_bound >= 0.5
        below_bound_rate = mean_rate(below_bound_mask)
        info[f"{prefix}/below_bound_rate"] = below_bound_rate
        info[f"{prefix}/rates/below_bound"] = below_bound_rate
        if failure_mask is not None:
            info[f"{prefix}/conditioned/below_bound_given_failure"] = conditional_rate(below_bound_mask, failure_mask)

    if above_bound is not None:
        above_bound_mask = above_bound >= 0.5
        above_bound_rate = mean_rate(above_bound_mask)
        info[f"{prefix}/above_bound_rate"] = above_bound_rate
        info[f"{prefix}/rates/above_bound"] = above_bound_rate
        if failure_mask is not None:
            info[f"{prefix}/conditioned/above_bound_given_failure"] = conditional_rate(above_bound_mask, failure_mask)

    if horizontal_out_of_bound is not None:
        horizontal_out_of_bound_mask = horizontal_out_of_bound >= 0.5
        horizontal_out_of_bound_rate = mean_rate(horizontal_out_of_bound_mask)
        info[f"{prefix}/horizontal_out_of_bound_rate"] = horizontal_out_of_bound_rate
        info[f"{prefix}/rates/horizontal_out_of_bound"] = horizontal_out_of_bound_rate
        if failure_mask is not None:
            info[f"{prefix}/conditioned/horizontal_out_of_bound_given_failure"] = conditional_rate(
                horizontal_out_of_bound_mask, failure_mask
            )

    if truncated is not None:
        time_limit_mask = truncated >= 0.5
        time_limit_rate = mean_rate(time_limit_mask)
        info[f"{prefix}/time_limit_rate"] = time_limit_rate
        info[f"{prefix}/rates/time_limit"] = time_limit_rate

    if deadlock is not None:
        deadlock_mask = deadlock >= 0.5
        deadlock_rate = mean_rate(deadlock_mask)
        info[f"{prefix}/deadlock_rate"] = deadlock_rate
        info[f"{prefix}/rates/deadlock"] = deadlock_rate
        if failure_mask is not None:
            info[f"{prefix}/deadlock_rate_on_failure"] = conditional_rate(deadlock_mask, failure_mask)
            info[f"{prefix}/conditioned/deadlock_given_failure"] = conditional_rate(deadlock_mask, failure_mask)
        if collision_mask is not None:
            info[f"{prefix}/conditioned/deadlock_given_collision"] = conditional_rate(deadlock_mask, collision_mask)
        if time_limit_mask is not None:
            info[f"{prefix}/conditioned/deadlock_given_time_limit"] = conditional_rate(deadlock_mask, time_limit_mask)

    if stuck_steps is not None:
        deadlock_steps = stuck_steps.float().mean().item()
        info[f"{prefix}/deadlock_steps"] = deadlock_steps
        info[f"{prefix}/deadlock/steps_mean"] = deadlock_steps
        if episode_len is not None:
            deadlock_step_ratio = (stuck_steps.float() / episode_len.float().clamp_min(1.0)).mean().item()
            info[f"{prefix}/deadlock_step_ratio"] = deadlock_step_ratio
            info[f"{prefix}/deadlock/step_ratio"] = deadlock_step_ratio

    if stall is not None:
        stall_mask = stall >= 0.5
        stall_rate = mean_rate(stall_mask)
        info[f"{prefix}/stall_rate"] = stall_rate
        info[f"{prefix}/rates/stall"] = stall_rate
        if failure_mask is not None:
            info[f"{prefix}/conditioned/stall_given_failure"] = conditional_rate(stall_mask, failure_mask)
        if collision_mask is not None:
            info[f"{prefix}/conditioned/stall_given_collision"] = conditional_rate(stall_mask, collision_mask)

    if stall_steps is not None:
        stall_steps_mean = stall_steps.float().mean().item()
        info[f"{prefix}/stall_steps"] = stall_steps_mean
        info[f"{prefix}/stall/steps_mean"] = stall_steps_mean
        if episode_len is not None:
            stall_step_ratio = (stall_steps.float() / episode_len.float().clamp_min(1.0)).mean().item()
            info[f"{prefix}/stall_step_ratio"] = stall_step_ratio
            info[f"{prefix}/stall/step_ratio"] = stall_step_ratio

    if reward_stall is not None:
        stall_reward_mean = reward_stall.float().mean().item()
        stall_reward_active_mask = reward_stall < -1e-6
        info[f"{prefix}/stall_reward_mean"] = stall_reward_mean
        info[f"{prefix}/stall/reward_mean"] = stall_reward_mean
        info[f"{prefix}/stall/active_rate"] = mean_rate(stall_reward_active_mask)
        if failure_mask is not None:
            info[f"{prefix}/stall/reward_on_failure"] = conditional_rate(reward_stall.float(), failure_mask)
        if collision_mask is not None:
            info[f"{prefix}/stall/reward_on_collision"] = conditional_rate(reward_stall.float(), collision_mask)

    if reward_vo is not None:
        vo_reward_mean = reward_vo.float().mean().item()
        vo_active_mask = reward_vo < -1e-6
        info[f"{prefix}/vo_reward_mean"] = vo_reward_mean
        info[f"{prefix}/vo/reward_mean"] = vo_reward_mean
        info[f"{prefix}/vo/active_rate"] = mean_rate(vo_active_mask)
        if failure_mask is not None:
            info[f"{prefix}/vo/reward_on_failure"] = conditional_rate(reward_vo.float(), failure_mask)
        if collision_mask is not None:
            info[f"{prefix}/vo/reward_on_collision"] = conditional_rate(reward_vo.float(), collision_mask)

    if vo_risk is not None:
        vo_risk_mean = vo_risk.float().mean().item()
        vo_risk_active_mask = vo_risk > 1e-6
        info[f"{prefix}/vo_risk_mean"] = vo_risk_mean
        info[f"{prefix}/vo/risk_mean"] = vo_risk_mean
        info[f"{prefix}/vo/risk_active_rate"] = mean_rate(vo_risk_active_mask)
        if failure_mask is not None:
            info[f"{prefix}/vo/risk_on_failure"] = conditional_rate(vo_risk.float(), failure_mask)
        if collision_mask is not None:
            info[f"{prefix}/vo/risk_on_collision"] = conditional_rate(vo_risk.float(), collision_mask)

    if vo_warmup is not None:
        vo_warmup_mean = vo_warmup.float().mean().item()
        info[f"{prefix}/vo_warmup"] = vo_warmup_mean
        info[f"{prefix}/vo/warmup"] = vo_warmup_mean

    if collision_mask is not None and deadlock_mask is not None:
        info[f"{prefix}/conditioned/collision_given_deadlock"] = conditional_rate(collision_mask, deadlock_mask)
    if time_limit_mask is not None and deadlock_mask is not None:
        info[f"{prefix}/conditioned/time_limit_given_deadlock"] = conditional_rate(time_limit_mask, deadlock_mask)
    if success_mask is not None and deadlock_mask is not None:
        info[f"{prefix}/conditioned/success_given_deadlock"] = conditional_rate(success_mask, deadlock_mask)
        info[f"{prefix}/conditioned/success_given_no_deadlock"] = conditional_rate(success_mask, ~deadlock_mask)

    return info


def resolve_eval_style(cfg):
    style = cfg.get("eval_style", "opposite_crossing_eval") if hasattr(cfg, "get") else "opposite_crossing_eval"
    style = str(style)

    if style in ("standard_eval", "standard"):
        return "standard", "standard_eval"
    if style in (
        "opposite_crossing_eval",
        "opposite_crossing",
        "random_crossing_eval",
        "random_crossing",
        "random",
    ):
        return "opposite_crossing", "opposite_crossing_eval"
    if style in ("wall_crossing_eval", "wall_crossing"):
        return "wall_crossing", "wall_crossing_eval"
    raise ValueError(
        f"Unknown eval_style={style}. Expected 'opposite_crossing_eval', "
        "'wall_crossing_eval', or 'standard_eval'."
    )


@torch.no_grad()
def evaluate(
    env,
    policy,
    cfg,
    seed: int=0, 
    exploration_type: ExplorationType=ExplorationType.MEAN,
    prefix: str="eval",
    eval_task_mode: str=None,
):
    base_env = getattr(env, "base_env", env)
    prev_task_mode = getattr(base_env, "eval_task_mode", None)
    prev_training = getattr(base_env, "training", None)

    env.enable_render(True)
    if eval_task_mode is not None and hasattr(base_env, "set_eval_task_mode"):
        base_env.set_eval_task_mode(eval_task_mode)
    if hasattr(base_env, "eval"):
        base_env.eval()
    env.eval()
    env.set_seed(seed)

    render_callback = RenderCallback(interval=2)
    
    with set_exploration_type(exploration_type):
        trajs = env.rollout(
            max_steps=env.max_episode_length,
            policy=policy,
            callback=render_callback,
            auto_reset=True,
            break_when_any_done=False,
            return_contiguous=False,
        )
    done = trajs.get(("next", "done")) 
    first_done = torch.argmax(done.long(), dim=1).cpu() # idx of first done will be return for each trajs

    def take_first_episode(tensor: torch.Tensor):
        indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
        return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

    traj_stats = {
        k: take_first_episode(v)
        for k, v in trajs[("next", "stats")].cpu().items()
    }

    info = summarize_episode_stats(traj_stats, prefix=prefix)

    # log video
    recording = wandb.Video(
        render_callback.get_video_array(axes="t c h w"), 
        fps=0.5 / (cfg.sim.dt * cfg.sim.substeps), 
        format="mp4"
    )
    info[f"{prefix}/recording"] = recording
    if prefix == "eval":
        info["recording"] = recording

    env.enable_render(not cfg.headless)
    if prev_task_mode is not None and hasattr(base_env, "set_eval_task_mode"):
        base_env.set_eval_task_mode(prev_task_mode)
    if prev_training is not None:
        if hasattr(base_env, "train"):
            base_env.train(prev_training)
        if hasattr(env, "train"):
            env.train(prev_training)
    else:
        env.train()
    env.reset()

    return info


def vec_to_new_frame(vec, goal_direction):
    if (len(vec.size()) == 1):
        vec = vec.unsqueeze(0)
    # print("vec: ", vec.shape)

    # goal direction x
    goal_direction_x = goal_direction / goal_direction.norm(dim=-1, keepdim=True)
    z_direction = torch.tensor([0, 0, 1.], device=vec.device)
    
    # goal direction y
    goal_direction_y = torch.cross(z_direction.expand_as(goal_direction_x), goal_direction_x)
    goal_direction_y /= goal_direction_y.norm(dim=-1, keepdim=True)
    
    # goal direction z
    goal_direction_z = torch.cross(goal_direction_x, goal_direction_y)
    goal_direction_z /= goal_direction_z.norm(dim=-1, keepdim=True)

    n = vec.size(0)
    if len(vec.size()) == 3:
        vec_x_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_x.view(n, 3, 1)) 
        vec_y_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_z.view(n, 3, 1))
    else:
        vec_x_new = torch.bmm(vec.view(n, 1, 3), goal_direction_x.view(n, 3, 1))
        vec_y_new = torch.bmm(vec.view(n, 1, 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, 1, 3), goal_direction_z.view(n, 3, 1))

    vec_new = torch.cat((vec_x_new, vec_y_new, vec_z_new), dim=-1)

    return vec_new


def vec_to_world(vec, goal_direction):
    world_dir = torch.tensor([1., 0, 0], device=vec.device).expand_as(goal_direction)
    
    # directional vector of world coordinate expressed in the local frame
    world_frame_new = vec_to_new_frame(world_dir, goal_direction)

    # convert the velocity in the local target coordinate to the world coodirnate
    world_frame_vel = vec_to_new_frame(vec, world_frame_new)
    return world_frame_vel


def construct_input(start, end):
    input = []
    for n in range(start, end):
        input.append(f"{n}")
    return "(" + "|".join(input) + ")"
